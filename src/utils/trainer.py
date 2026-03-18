import sys, os, copy, time, csv
import numpy as np
from datetime import datetime
import pickle
import torch.nn.functional as F
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from colorama import Back, Fore, Style
from .core_tools import (
    get_optim,
    print_init_msg,
    load_model,
    get_dataset,
    compute_loss,
    get_evaluator,
    Collator,
    EarlyStopping
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.metrics import roc_auc_score, accuracy_score
import random


class Trainer():
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.device = args.device
        self.missing_type = args.missing_type
        self.task = args.dataset
        self.save_path = args.save_path
        self.model = load_model(
            missing_type=args.missing_type,
            task_id = self.task,
            device = args.device,
            max_text_len = args.max_text_len,
            max_image_len = args.max_image_len,
            model=args.model,
            backbone=args.backbone,
            vilt_weights=args.vilt_weights,
            prompt_position=args.prompt_position,
            prompt_length=args.prompt_length,
            dropout_rate=args.dropout_rate)
        self.model.to(self.device)


        train_dataset = get_dataset(
            dataset_name=self.dataset,
            split='train',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        valid_dataset = get_dataset(
            dataset_name=self.dataset,
            split='valid',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        test_dataset = get_dataset(
            dataset_name=self.dataset,
            split='test',
            dataset=args.dataset,
            missing_type=args.missing_type,
            missing_rate=args.missing_rate,
            max_text_len=args.max_text_len,
            max_image_len=args.max_image_len,
            k=args.k
        )

        # 自定义批处理函数，将原始样本列表统一格式化、编码成张量，供模型直接使用
        collator = Collator(max_text_len=args.max_text_len)

        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=True
        )

        self.valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=False
        )

        self.test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            shuffle=False
        )


        self.optimizer, self.scheduler = get_optim(
            max_steps=len(self.train_data_loader) * self.epochs,
            model=self.model,
            name=args.name,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_warmup=args.use_warmup,
            warmup_rate=args.warmup_rate
        )

        self.evaluator = get_evaluator(self.dataset, args.device)
        self.early_stopper = EarlyStopping(patience=args.patience, delta=0.0, path=os.path.join(self.save_path, "best_model.pth"))

    def run(self):

        best_score = -1.0  # 当前验证集上最佳 AUROC
        best_epoch = -1
        REFRESH_EVERY = 5
        pair_ent_id = None

        # 课程表参数（可按需改）
        ratio_start = 0.20  # warmup 之后的起始比例
        ratio_end = 0.30  # 最终比例
        grow_epochs = 5  # 用多少个 epoch 从 start 线性涨到 end

        for epoch in range(self.epochs):
            print(f"{Fore.RED}Current Epoch: {epoch + 1}{Style.RESET_ALL}")

            # 是否需要刷新：首次或每隔 REFRESH_EVERY 个 epoch
            need_refresh = (pair_ent_id is None) or (epoch % REFRESH_EVERY == 0)
            if need_refresh:
                pair_ent_id = self._ranked_missing_samples()

            # 线性课程
            progress = epoch
            t = min(progress / max(1, grow_epochs), 1.0)
            ratio = ratio_start + (ratio_end - ratio_start) * t
            k = max(1, int(len(pair_ent_id) * ratio)) if pair_ent_id else 0
            reliable_ids = set(id_ for _, id_ in pair_ent_id[:k]) if k > 0 else set()
            print(f"{Fore.RED}Ratio={ratio:.2f}, Reliable_ids={len(reliable_ids)}{Style.RESET_ALL}")
            self._train(reliable_ids)

            # ========== 验证 and 早停 ==========
            val_metrics = self._valid(current_epoch=epoch + 1)
            val_score = float(val_metrics['auroc'])

            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
            stop = self.early_stopper(val_score, self.model)

            if stop:
                print(f"{Fore.RED}Early stopping at epoch {epoch + 1}. Best AUROC={best_score:.4f} @ epoch {best_epoch}.{Style.RESET_ALL}")
                break

        # ===== 训练结束：加载最佳权重，进行测试=====
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "best_model.pth"), map_location=self.device))
        self._test()

    def _train(self, reliable_ids=None):
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_data_loader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}", desc='Training')
        count_zero, count_inside, count_project = 0, 0, 0

        for batch in pbar:
            inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
            labels = inputs.pop('label')
            batch_ids = inputs.pop('id')
            missing_mask = inputs['missing_mask']
            preds = self.model(**inputs)

            # 1. 计算每个样本损失
            if self.missing_type == "Text" or self.missing_type == "Image":
                per_loss = compute_loss(preds, labels, reduction='none')
                idx_complete = (missing_mask == 1)
                idx_missing = (missing_mask == 0)

                if reliable_ids is None or len(reliable_ids) == 0:
                    idx_complete_eff = idx_complete
                    idx_missing_eff = idx_missing
                    num_promoted = 0
                else:
                    in_set = [i in reliable_ids for i in batch_ids]
                    reliable_mask = torch.tensor(in_set, device=self.device, dtype=torch.bool)
                    idx_complete_eff = idx_complete | (idx_missing & reliable_mask)
                    idx_missing_eff = idx_missing & (~reliable_mask)
                    num_promoted = (idx_missing & reliable_mask).sum().item() # 单个batch中由补全样本转为完备样本的个数

                LC = per_loss[idx_complete_eff].mean() if idx_complete_eff.any() else None
                LM = per_loss[idx_missing_eff].mean() if idx_missing_eff.any() else None
                loss_list.append(((0.0 if LC is None else float(LC)) + (0.0 if LM is None else float(LM))))

            elif self.missing_type == "Both":
                per_loss = compute_loss(preds, labels, reduction='none')
                idx_complete = (missing_mask == 2)
                idx_missing_image = (missing_mask == 1)
                idx_missing_text = (missing_mask == 0)
                idx_missing_union = idx_missing_image | idx_missing_text

                if reliable_ids is None or len(reliable_ids) == 0:
                    idx_complete_eff = idx_complete
                    idx_missing_eff = idx_missing_union
                    num_promoted = 0
                else:
                    in_set = [i in reliable_ids for i in batch_ids]
                    reliable_mask = torch.tensor(in_set, device=self.device, dtype=torch.bool)
                    idx_complete_eff = idx_complete | (idx_missing_union & reliable_mask)
                    idx_missing_eff = idx_missing_union & (~reliable_mask)
                    num_promoted = (idx_missing_union & reliable_mask).sum().item() # 单个batch中由补全样本转为完备样本的个数

                LC = per_loss[idx_complete_eff].mean() if idx_complete_eff.any() else None
                LM = per_loss[idx_missing_eff].mean() if idx_missing_eff.any() else None
                loss_list.append(((0.0 if LC is None else float(LC)) + (0.0 if LM is None else float(LM))))

            # 2. 梯度拆分，求gC和gM
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer.zero_grad(set_to_none=True)
            if LC is not None:
                LC.backward(retain_graph=True)
            gC = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in trainable]

            self.optimizer.zero_grad(set_to_none=True)
            if LM is not None:
                LM.backward()
            gM = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in trainable]

            # 3. Anchor构建
            with torch.no_grad():
                # 3.1 计算 gC 的整体范数（把各层梯度看成一个拼接大向量）
                gC_sqsum = torch.tensor(0.0, device=self.device)
                for gc in gC:
                    gC_sqsum += (gc.float() ** 2).sum()
                a_norm = gC_sqsum.sqrt().clamp_min(1e-12)

                # 3.2 归一化得到主轴各参数分量（与 trainable 一一对应）
                a_t = [gc / a_norm for gc in gC]

            # 4. 锥域投影：把 gM 投影到以 a_t 为轴、阈值 cos>=tau 的锥域内
            tau = 0.7
            eps = 1e-12

            # 4.1 计算 cos(gM, a_t)
            dot = torch.tensor(0.0, device=self.device)
            gm_sq = torch.tensor(0.0, device=self.device)
            for gm, ah in zip(gM, a_t):
                dot += (gm * ah).float().sum()
                gm_sq += (gm.float() ** 2).sum()
            gm_norm = gm_sq.sqrt().clamp_min(eps)
            cos_gm_a = (dot / gm_norm).clamp(-1.0, 1.0) # 一个值

            # 4.2 三种情况
            if cos_gm_a <= 0:  # 反向：置零
                gM_proj = [torch.zeros_like(gm) for gm in gM]
                count_zero += 1

            elif cos_gm_a >= tau:  # 已在锥内：不动
                gM_proj = gM
                count_inside += 1

            else:
                # 投影到锥边界
                # 平行分量 g_parallel = (gM·a)a；注意 a_t 是单位向量
                g_par = [dot * ah for ah in a_t]

                # 垂直分量 g_perp = gM - g_parallel
                g_perp = [gm - gp for gm, gp in zip(gM, g_par)]

                # 允许上限 t_max = sqrt(1 - tau^2) / tau
                t_max = torch.sqrt(torch.tensor(1.0, device=self.device) - tau ** 2) / (tau + eps)

                # ||g_parallel|| = |gM·a| （因为 a 是单位向量）
                gpar_norm = torch.abs(dot)

                # ||g_perp||
                gperp_sq = torch.tensor(0.0, device=self.device)
                for gp in g_perp:
                    gperp_sq += (gp.float() ** 2).sum()
                gperp_norm = gperp_sq.sqrt().clamp_min(eps)

                # s = t_max * ||g_parallel|| / (||g_perp|| + eps)
                s = t_max * gpar_norm / (gperp_norm + eps)

                # g̃ = g_parallel + s * g_perp
                gM_proj = [gp_par + s * gp_perp for gp_par, gp_perp in zip(g_par, g_perp)]
                count_project += 1

            # 5. 写回总梯度并更新：g_total = gC + g̃M
            self.optimizer.zero_grad(set_to_none=True)
            for p, gc, gm in zip(trainable, gC, gM_proj):
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(gc + gm)
            self.optimizer.step()
            self.scheduler.step()

        print(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list):.4f}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Stats: Zero={count_zero}, Inside={count_inside}, Project={count_project}{Style.RESET_ALL}")

    def _valid(self, current_epoch=None):
        self.model.eval()
        loss_list = []

        missing_cnt, complete_cnt = 0, 0
        eva_all = copy.deepcopy(self.evaluator); eva_all.reset()
        eva_complete = copy.deepcopy(self.evaluator); eva_complete.reset()
        eva_missing = copy.deepcopy(self.evaluator); eva_missing.reset()

        self.test_data_loader

        pbar = tqdm(self.valid_data_loader, bar_format=f"{Fore.YELLOW}{{l_bar}}{{bar}}{{r_bar}}", desc='Validating')

        with torch.no_grad():
            for batch in pbar:
                missing_mask = batch["missing_mask"].to(self.device)
                missing_cnt += int((missing_mask == 0).sum())
                complete_cnt += int((missing_mask == 1).sum())

                # ---- 前向 ----
                inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
                labels = inputs.pop('label')
                batch_ids = inputs.pop('id')
                preds = self.model(**inputs)
                loss = compute_loss(preds, labels)
                loss_list.append(loss.item())

                # ---- 更新 evaluator ---
                eva_all.update(preds=preds, labels=labels)

                idx_complete = (missing_mask == 1).nonzero(as_tuple=True)[0]
                idx_missing = (missing_mask == 0).nonzero(as_tuple=True)[0]
                if idx_complete.numel():
                    eva_complete.update(preds=preds[idx_complete], labels=labels[idx_complete])
                if idx_missing.numel():
                    eva_missing.update(preds=preds[idx_missing], labels=labels[idx_missing])

        # ---- 计算指标 ----
        metrics_all = eva_all.compute()
        metrics_complete = eva_complete.compute() if complete_cnt > 0 else {"auroc": float("nan"), "acc": float("nan")}
        metrics_missing = eva_missing.compute() if missing_cnt > 0 else {"auroc": float("nan"), "acc": float("nan")}

        print(f"{Fore.YELLOW}Valid: Loss: {np.mean(loss_list):.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (all)      {metrics_all['auroc']:.4f} | ACC (all)      {metrics_all['acc']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (complete) {metrics_complete['auroc']:.4f} | ACC (complete) {metrics_complete['acc']:.4f}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}AUROC (missing)  {metrics_missing['auroc']:.4f} | ACC (missing)  {metrics_missing['acc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (all)      {metrics_all['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (complete) {metrics_complete['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}AUROC (missing)  {metrics_missing['auroc']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Valid: 完整样本 {complete_cnt}，缺失样本 {missing_cnt}{Style.RESET_ALL}")

        # ---- 保存指标 ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(f"/data/gzh/MissingWork/MyWork/src/metrics/metrics_{timestamp}.csv")

        if not log_path.exists():
            with open(log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "auroc_all", "auroc_complete", "auroc_missing"])

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch if current_epoch is not None else -1,
                metrics_all["auroc"],
                metrics_complete["auroc"],
                metrics_missing["auroc"],
            ])

        return metrics_all

    def _test(self):
        self.model.eval()

        eva = copy.deepcopy(self.evaluator); eva.reset()

        pbar = tqdm(self.test_data_loader, bar_format=f"{Fore.RED}{{l_bar}}{{bar}}{{r_bar}}", desc='Testing')

        with torch.no_grad():
            for batch in pbar:
                # ---- 前向 ----
                inputs = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
                labels = inputs.pop('label')
                batch_ids = inputs.pop('id')
                preds = self.model(**inputs)

                # ---- 更新 evaluator ---
                eva.update(preds=preds, labels=labels)

        # ---- 计算指标 ----
        metrics_all = eva.compute()
        print(f"{Fore.RED}AUROC (all)      {metrics_all['auroc']:.4f}{Style.RESET_ALL}")

    def _ranked_missing_samples(self):
        """
        返回全局熵最低前 ratio 比例的补全集样本 id 集合。
        需要：Dataset 在 __getitem__ 里返回 'id'（全局索引）。
        """
        self.model.eval()
        pbar = tqdm(self.train_data_loader, bar_format=f"{Fore.RED}{{l_bar}}{{bar}}{{r_bar}}", desc='Selecting Reliable Samples')

        pair_ent_id = []
        with torch.no_grad():
            for batch in pbar:
                ids = batch['id']
                missing_mask = batch['missing_mask'].to(self.device)

                inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()
                          if k not in ['label', 'id']}

                preds = self.model(**inputs)
                probs = torch.softmax(preds, dim=-1).clamp_min(1e-12)
                ent = -(probs * probs.log()).sum(dim=-1)  # [B] 熵，越小越可靠

                miss_mask = (missing_mask == 0)
                if miss_mask.any():
                    sel_idx = torch.nonzero(miss_mask, as_tuple=False).squeeze(1).cpu().tolist()
                    ids_sel = [ids[i] for i in sel_idx]
                    ents_sel = ent[miss_mask].detach().cpu().tolist()
                    pair_ent_id.extend(zip(ents_sel, ids_sel))

        pair_ent_id.sort(key=lambda x: x[0])

        return pair_ent_id

















