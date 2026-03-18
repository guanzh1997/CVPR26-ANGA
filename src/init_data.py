import sys, os
from utils import (
    init_data_hatememes,
    MemoryBankGenerator,
    MCR)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import itertools

def main():
    pd.set_option('future.no_silent_downcasting', True)

    # 1️⃣ 生成train/valid/test.pkl文件(8500/500/1000, 4), columns name: item_id, img, label, text
    # init_data_hatememes()
    # df_train = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/train.pkl')
    # df_valid = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/valid.pkl')
    # df_test = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/test.pkl')
    # print(df_train.shape, df_valid.shape, df_test.shape)
    # sys.exit(0)

    # 2️⃣ 构建Memory
    # memory_bank_generator = MemoryBankGenerator()
    # memory_bank_generator.run()
    # image_count = sum(len(files) for _, _, files in os.walk("/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/image"))
    # text_count= sum(len(files) for _, _, files in os.walk("/data/gzh/MissingWork/MyWork/dataset/memory_bank/hatememes/text"))
    # print(image_count, text_count)
    # sys.exit(0)

    # 3️⃣ 构建检索列表，生成train/valid/test.pkl文件(8500/500/1000, 12)
    # mcr = MCR()
    # mcr.run()
    # df_train_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/train.pkl')
    # df_valid_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/valid.pkl')
    # df_test_ = pd.read_pickle('/data/gzh/MissingWork/MyWork/dataset/hatememes/test.pkl')
    # print(df_train_.shape, df_valid_.shape, df_test_.shape)
    # sys.exit(0)


if  __name__ == '__main__':
    main()