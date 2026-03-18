import argparse
import sys
from utils import (
    seed_init,
    generate_missing_table,
    Trainer)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def main():

    if True:
        parser = argparse.ArgumentParser(description="Model and Training Configuration")

        # one modality missing: 0 missing, 1 not missing
        # two modality missing: 0 text missing, 1 image missing, 2 not missing

        # Model parameters
        parser.add_argument('--model', type=str, default="ANGA")
        parser.add_argument('--backbone', type=str, default="vilt")
        parser.add_argument('--vilt_weights', type=str, default="src/model/vilt/weights/mlm")
        parser.add_argument('--prompt_position', type=int, default=0)
        parser.add_argument('--prompt_length', type=int, default=1)
        parser.add_argument('--dropout_rate', type=float, default=0.2)

        # Data parameters
        parser.add_argument('--dataset', type=str, default="hatememes", choices=["hatememes", "mmimdb", "food101"])
        parser.add_argument('--missing_type', type=str, default="Text", choices=["Both", "Text", "Image"])
        parser.add_argument('--missing_rate', type=float, default=0.7)
        parser.add_argument('--max_text_len', type=int, default=128)
        parser.add_argument('--max_image_len', type=int, default=145)
        parser.add_argument('--k', type=int, default=5)

        # Optimizer parameters
        parser.add_argument('--name', type=str, default="AdamW")
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=5e-5)
        parser.add_argument('--use_warmup', type=bool, default=True)
        parser.add_argument('--warmup_rate', type=float, default=0.1)

        # Global parameters
        parser.add_argument('--device', type=str, default="cuda:4")
        parser.add_argument('--seed', type=int, default=2024)
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--save_path', type=str, default="/data/gzh/MissingWork/MyWork/src/checkpoints")
        parser.add_argument('--regenerate_missing_table', type=bool, default=False)

        args = parser.parse_args()

    pd.set_option('future.no_silent_downcasting', True) # 控制 Pandas 中的数据类型自动降级行为的选项
    seed_init(args.seed)

    # single：0缺失：1没有缺失
    # both: 0缺失文本；1缺失图像；2没有缺失
    if args.regenerate_missing_table:
        data_para = {
            'dataset': args.dataset,
            'missing_type': args.missing_type,
            'missing_rate': args.missing_rate,}
        generate_missing_table(**data_para)
        sys.exit(0)

    # # 加载并查看 missing_table.pkl 的内容
    # file_path = '/data/gzh/MissingWork/MyWork/dataset/missing_table/both/hatememes/missing_table.pkl'  # 修改为你的文件路径
    # df = pd.read_pickle(file_path)
    # total_items = df['item_id'].nunique()
    # missing_counts = df['missing_mask_7'].value_counts()
    # # 输出结果
    # print(f"Total number of unique item_ids: {total_items}")
    # print(f"Missing data (0): {missing_counts.get(0, 0)}")
    # print(f"Missing data (1): {missing_counts.get(1, 0)}")
    # print(f"Missing data (2): {missing_counts.get(2, 0)}")
    # print(df[:10])
    # print(df.shape)
    # sys.exit(0)

    trainer = Trainer(args)
    trainer.run()

if  __name__ == '__main__':
    main()