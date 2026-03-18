import sys

from PIL import  Image
import numpy as np
import torch.utils.data
import os
import pandas as pd
from PIL import Image

class HatememesDataset(torch.utils.data.Dataset):
    def __init__(self, split, max_text_len,  missing_type, missing_rate, k, **kargs):
        super().__init__()
        # 1️⃣ 修改
        dataframe = pd.read_pickle(os.path.join('/data/gzh/MissingWork/MyWork/dataset/hatememes', f'{split}.pkl'))
        if missing_type == "Image" or missing_type == "Text":
            missing_table = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/missing_table/single/hatememes/missing_table.pkl')
        elif missing_type == "Both":
            missing_table = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/missing_table/both/hatememes/missing_table.pkl')

        # DataFrame: (8500/1000,13)
        dataframe = pd.merge(dataframe, missing_table, on='item_id')
        self.k = k
        self.missing_type = missing_type
        self.max_text_len = max_text_len
        self.id_list = dataframe['item_id'].tolist()
        self.text_list = dataframe['text'].tolist()
        self.label_list = dataframe['label'].tolist()
        self.i2i_list = dataframe['i2i_id_list'].tolist()
        self.t2t_list = dataframe['t2t_id_list'].tolist()
        self.i2i_r_l_list_list = dataframe['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = dataframe['t2t_label_list'].tolist()
        self.missing_mask_list = dataframe[f'missing_mask_{int(10 * missing_rate)}'].tolist()

    def __getitem__(self, index):
        k = self.k
        text = self.text_list[index]
        image = Image.open(fr'/data/gzh/MissingWork/MyWork/dataset/hatememes/image/{self.id_list[index]}.png').convert("RGB")
        r_t_list = []
        r_i_list = []

        # 2️⃣ 修改
        if self.missing_type == "Text" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]

            # 取出前top-k个与该样本图像相似度最高的item_id，并提取对应的图像特征和文本特征
            for i in i2i_list[:k]:
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())

            # 提取全部标签
            r_l_list = self.i2i_r_l_list_list[index]

        elif self.missing_type == "Image" and self.missing_mask_list[index] == 0:
            t2t_list = self.t2t_list[index]

            for i in t2t_list[:k]:
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]
            
        elif self.missing_type == "Text" or self.missing_type == "Image" and self.missing_mask_list[index] == 1:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            
            for i in t2t_list[:k]:
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]
        
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 0:
            text = "I love deep learning" * 1024
            i2i_list = self.i2i_list[index]
            for i in i2i_list[:k]:
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            r_l_list = self.i2i_r_l_list_list[index]
        
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 1:
            t2t_list = self.t2t_list[index]
            for i in t2t_list[:k]:
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            r_l_list = self.t2t_r_l_list_list[index]
        
        elif self.missing_type == "Both" and self.missing_mask_list[index] == 2:
            i2i_list = self.i2i_list[index]
            t2t_list = self.t2t_list[index]

            for i in i2i_list[:k]:
                r_i = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image/{i}.npy')
                r_i_list.append(r_i.tolist())
            
            for i in t2t_list[:k]:
                r_t = np.load(fr'/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text/{i}.npy')
                r_t_list.append(r_t.tolist())
            
            r_l_list = self.i2i_r_l_list_list[index]
        
        return {
            "image": image,                                         # 图像特征（全）
            "text": text,                                           # 文本特征（全）若缺失则填充
            "label": self.label_list[index],                        # 标签（全）
            "r_t_list": r_t_list,                                   # 检索到的文本（Top-K）---若缺失，和下面成对，否则是全局的不一定成对
            "r_i_list": r_i_list,                                   # 检索到的图像（Top-K）---若缺失，和上面成对，否则是全局的不一定成对
            "missing_mask": self.missing_mask_list[index],          # 0：缺失；1：不缺失
            "r_l_list": r_l_list[:k],                               # 检索到Top-K个实例的对应标签
            "id": self.id_list[index]
        }
    def __len__(self):
        return len(self.text_list)