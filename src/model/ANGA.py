import torch
from torch import nn
import torch.nn.functional as F
from .vilt import ViltModel
from .modules import MMG, CAP
# from einops import rearrange

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class ANGA(torch.nn.Module):
    def __init__(self,
                 vilt: ViltModel,
                 task_id: str,
                 max_text_len: int,
                 max_image_len: int,
                 missing_type: str,
                 device: str,
                 prompt_position: int,
                 prompt_length: int,
                 dropout_rate: float,
                 hs=768,
                 **kargs):
        super(ANGA, self).__init__()
        self.device = device
        self.max_text_len = max_text_len
        self.missing_type = missing_type
        self.task_id = task_id
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position
        self.hs = hs

        if task_id == "hatememes":
            cls_num = 2
        elif task_id == "food101":
            cls_num = 101
        elif task_id == "mmimdb":
            cls_num = 23

        # freeze the pretrained multi-modal transformer
        self.freeze()

        # define training component
        self.pooler = vilt.pooler

        # define the MMG for completion of missing information
        if missing_type == "Text":
            self.MMG = MMG(n = max_text_len, d = hs,dropout_rate=dropout_rate)
        elif missing_type == "Image":
            self.MMG = MMG(n = max_image_len, d = hs,dropout_rate=dropout_rate)
        elif missing_type == "Both":
            self.MMG_t = MMG(n = max_text_len, d = hs,dropout_rate=dropout_rate)
            self.MMG_i = MMG(n = max_image_len, d = hs,dropout_rate=dropout_rate)

        # define the dynamic prompt
        self.dynamic_prompt = CAP(prompt_length=prompt_length)

        self.classifier = nn.Linear(768, cls_num)
        self.classifier.apply(init_weights)

        # define the classifier
        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hs))
        # self.classifier = nn.Sequential(
        #         nn.Linear(hs * 2, hs * 2),
        #         nn.LayerNorm(hs * 2),
        #         nn.GELU(),
        #         nn.Linear(hs * 2, hs),
        #     )
        # self.classifier.apply(init_weights)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False




    def forward(self,
                input_ids: torch.Tensor,        # 文本 token ids (64,128)
                pixel_values: torch.Tensor,     # 图像像素张量 (64,3,384,384)
                pixel_mask: torch.Tensor,       # 图像掩码 (64,384,384) 好像全是1
                token_type_ids: torch.Tensor,   # 文本 segment ids (64,128) 好像全是0
                attention_mask: torch.Tensor,   # 文本 attention mask (64,128)
                r_t_list: torch.Tensor,         # 检索的文本向量 (64,5,128,768)
                r_i_list: torch.Tensor,         # 检索的图像向量 (64,5,145,768)
                r_l_list: torch.Tensor,         # 检索的标签 (64,5)
                missing_mask = None,            # 缺失掩码 (64,) 若没缺失，则全为1
                image_token_type_idx=1):

        # embedding: (64,273,768); attention_mask: (64,273)
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)

        # 1️⃣. 得到图文表征，mask后再补全
        # text_emb: (64,128,768); image_emb: (64,145,768)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]

        if self.missing_type == "Text":
            recovered_t = self.MMG(r_t_list) # (64,128,768) # 仅检索平均
            missing_mask_t = missing_mask.view(-1, 1, 1).expand(-1, 128, self.hs) # (64,128,768)
            text_emb = text_emb * missing_mask_t + recovered_t * (1-missing_mask_t) # (64,128,768)

        elif self.missing_type == "Image":
            recovered_i = self.MMG(r_i_list) # (64,145,768) # 仅检索平均
            missing_mask_i = missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs) # (64,145,768)
            image_emb = image_emb * missing_mask_i + recovered_i * (1-missing_mask_i) # (64,145,768)

        elif self.missing_type == "Both":
            recovered_t = self.MMG_t(r_t_list)
            recovered_i = self.MMG_i(r_i_list)
            t_missing_mask = [0 if i == 0 else 1 for i in missing_mask]
            i_missing_mask = [0 if i == 1 else 1 for i in missing_mask]
            t_missing_mask = torch.tensor(t_missing_mask).to(self.device)
            i_missing_mask = torch.tensor(i_missing_mask).to(self.device)
            missing_mask_t = t_missing_mask.view(-1, 1, 1).expand(-1, 128, self.hs)
            missing_mask_i = i_missing_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
            text_emb = text_emb * missing_mask_t + recovered_t * (1-missing_mask_t)
            image_emb = image_emb * missing_mask_i + recovered_i * (1-missing_mask_i)

        # 2️⃣. 构建提示
        t_prompt,i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb) # (64,1,768) (64,1,768)
        t_prompt = torch.mean(t_prompt, dim=1) # (64,768)
        i_prompt = torch.mean(i_prompt, dim=1) # (64,768)

        label_emb = self.label_enhanced[r_l_list] # (64,5,768)
        label_emb = torch.mean(label_emb, dim=1) # (64,768)
        label_emb = label_emb.view(-1, 1, self.hs) # (64,1,768)

        # 3️⃣. 模型训练
        output = torch.cat([text_emb, image_emb], dim=1)  # (64,273,768)
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                output = torch.cat([label_emb, t_prompt.unsqueeze(1), i_prompt.unsqueeze(1), output], dim=1) # (64,276,768)
                N = embedding.shape[0] # int: 64
                attention_mask = torch.cat([torch.ones(N, self.prompt_length*2+1).to(self.device), attention_mask], dim=1) # (64,276)
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0] # (64,276,768)
            else:
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0] # (64,276,768)

        output = self.layernorm(output) # (64,276,768)
        output = self.pooler(output) # (64,768)
        output = self.classifier(output) # (64,2)

        return output