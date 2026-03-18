# Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning

This repo is the official implementation of _Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning_ accepted by CVPR 2026. 

## Abstract

Vision-language pre-training (VLP) has achieved remarkable performance across diverse multimodal learning (MML) tasks. Recently, many efforts have focused on reconstructing missing modalities to improve the adaptability of VLP models in incomplete MML scenarios. However, these approaches overlook the learning imbalance under severe missing-modality conditions, i.e., the optimization process is dominated by reconstructed samples, thereby weakening complete-sample representations. In this paper, we propose a novel ANchor-guided Gradient Alignment (ANGA) framework to address this issue. Specifically, we first retrieve similar instances to reconstruct the missing modalities, thereby alleviating information deficiency. We then introduce an entropy-driven curriculum that progressively incorporates reliable reconstructed samples together with complete ones to form an optimization anchor, which guides gradient alignment to mitigate learning imbalance. Furthermore, we design a semantic-enhanced adapter that leverages the retrieved instances to generate dynamic prompts, further enhancing the robustness of the VLP model. Extensive experiments on widely used datasets demonstrate the superiority of ANGA over state-of-the-art (SOTA) baselines across various missing-modality scenarios.

## Framework

<img width="1232" alt="image" src="https://github.com/user-attachments/assets/0a7e7510-076d-4dd0-99cd-dcec59dc775e" />

## Missing Setting

We assume training set is fully available and define the missing rate $\eta$ % as the rate of modality-incomplete samples in the **test set**: (1) text/image missing with $\eta$ % indicates that there are $\eta$ % image-only/text-only instances and (1 - $\eta$ %) modality-complete instances. (2) both modalities missing with $\eta$ % indicates that there are $\frac{\eta}{2}$ % text-only instances, $\frac{\eta}{2}$ % image-only instances and (1 - $\eta$ %) modality-complete instances. We set **missing rate $\eta$ = 70 by default**. For training of each model (both ours and baselines), **we simulate the same 70 % missing rate to align model optimization well with test conditions,** but **allow each model to access the full modality information in the training set** to assist the training process. (e.g., for reconstruction models, we allow them to leverage the full modality information from training set as the reconstruction learning supervision signals). And **our memory bank is only constructed with the training and validation set** for a fair comparison (without any test data leakage).

## Environment Configuration

First, clone this repo:

```shell
git clone https://github.com/Jian-Lang/RAGPT.git

cd RAGPT
```

First, create a new conda env for RAGPT:

```shell
conda create -n RAGPT python=3.9
```

Next, activate this env and install the dependencies from the requirements.txt:

```shell
conda activate RAGPT

pip install -r requirements.txt
```

## Data Preparation

### MM-IMDb

First, download the dataset from this link: https://archive.org/download/mmimdb/mmimdb.tar.gz

Then, place the raw images in folder **dataset/mmimdb/image** and put the json files in folder **dataset/mmimdb/meta_data**.

### HateMemes

First, download the dataset from this link: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset

Then, place the raw images in folder **dataset/hatememes/image** and put the json files in folder **dataset/hatememes/metadata**.

Next, replace the **test.json** in metadata with **test_seen.json** downloaded from this link: https://www.kaggle.com/datasets/williamberrios/hateful-memes as the test.json downloaded from the prior website has no label information for evaluation. (Do not change other files, only replace the test.json with test_seen.json)

### Food101

First, download the dataset from this link: https://www.kaggle.com/datasets/gianmarco96/upmcfood101

Then, place the raw images in folder **dataset/food101/image** and put the csv files in folder **dataset/food101/meta_data**.

Note: The image folder follows the structure dataset/xxx/image/xxx.jpg|png|jpeg

## Code Running

### Dataset Initiation

Run the following script to init the dataset:

```shell
sh src/scripts/init_data.sh
```

### Training & Evaluation

Run the following script to training our model and evaluate the results:

```shell
sh src/scripts/eval.sh
```

All the parameters have the same meaning as describe in our paper and you can simply config them in **src/config/config.yaml** or in command line.


Run the following script to training baseline model and evaluate the results:

```shell
sh src/scripts/eval_baseline.sh
```

## Citation

If you find the code useful for your research, please give us a star ⭐⭐⭐ and consider citing:

```
@inproceedings{lang2025retrievalaugmented,
    author = {Lang, Jian and Cheng, Zhangtao and Zhong, Ting and Zhou, Fan},
    booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
    year = {2025},
    pages = {18035--18043},
    doi = {10.1609/aaai.v39i17.33984},
    title = {Retrieval-Augmented Dynamic Prompt Tuning for Incomplete Multimodal Learning},
}
```
