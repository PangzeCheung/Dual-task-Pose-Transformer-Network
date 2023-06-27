# Dual-task Pose Transformer Network
The source code for our paper "[
Pose Guided Person Image Generation via Dual-task Correlation and Affinity Learning](https://ieeexplore.ieee.org/abstract/document/10153661)“, Pengze Zhang, Lingxiao Yang, Jianhuang Lai, and Xiaohua Xie, TVCG 2023.
<img width="1148" alt="framework" src="https://user-images.githubusercontent.com/37894893/156797980-6387165c-3db8-48be-969f-011d3ecc3c05.png">

## Abstract

Pose Guided Person Image Generation (PGPIG) is the task of transforming a person's image from the source pose to a target pose. Existing PGPIG methods often tend to learn an end-to-end transformation between the source image and the target image, but do not seriously consider two issues: 1) the PGPIG is an ill-posed problem, and 2) the texture mapping requires effective supervision. In order to alleviate these two challenges, we propose a novel method by incorporating D ual-task P ose T ransformer N etwork and T exture A ffinity learning mechanism (DPTN-TA). To assist the ill-posed source-to-target task learning, DPTN-TA introduces an auxiliary task, i.e., source-to-source task, by a Siamese structure and further explores the dual-task correlation. Specifically, the correlation is built by the proposed Pose Transformer Module (PTM), which can adaptively capture the fine-grained mapping between sources and targets and can promote the source texture transmission to enhance the details of the generated images. Moreover, we propose a novel texture affinity loss to better supervise the learning of texture mapping. In this way, the network is able to learn complex spatial transformations effectively. Extensive experiments show that our DPTN-TA can produce perceptually realistic person images under significant pose changes. Furthermore, our DPTN-TA is not limited to processing human bodies but can be flexibly extended to view synthesis of other objects, i.e., faces and chairs, outperforming the state-of-the-arts in terms of both LPIPS and FID.


## Get Start

### 1) Requirement

* Python 3.7.9
* Pytorch 1.7.1
* torchvision 0.8.2
* CUDA 11.1
* NVIDIA A100 40GB PCIe

### 2) Data Preperation

Following **[PATN](https://github.com/tengteng95/Pose-Transfer)**, the dataset split files and extracted keypoints files can be obtained as follows:

**DeepFashion**


* Download the DeepFashion dataset (High-res) **[in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)**, and put them under the `./dataset/fashion` directory.

* Download train/test pairs and train/test keypoints annotations from **[Google Drive](https://drive.google.com/drive/folders/1qZDod3QDD7PaBxnNyHCuLBR7ftTSkSE1?usp=sharing)**, including **fasion-resize-pairs-train.csv, fasion-resize-pairs-test.csv, fasion-resize-annotation-train.csv, fasion-resize-annotation-train.csv, train.lst, test.lst**, and put them under the `./dataset/fashion` directory.

* Split the raw image into the training set (`./dataset/fashion/train`) and test set (`./dataset/fashion/test`):
``` bash
python data/generate_fashion_datasets.py
```

**Market1501**

* Download the Market1501 dataset from **[here](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)**. Rename **bounding_box_train** and **bounding_box_test** as **train** and **test**, and put them under the `./dataset/market` directory.

* Download train/test key points annotations from **[Google Drive](https://drive.google.com/drive/folders/1zzkimhX_D5gR1G8txTQkPXwdZPRcnrAx?usp=sharing)** including **market-pairs-train.csv, market-pairs-test.csv, market-annotation-train.csv, market-annotation-train.csv**. Put these files under the `./dataset/market` directory.

### 3) Train a model

**DeepFashion**
``` bash
python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 32 --gpu_id=0
```
**Market1501**

``` bash
python train.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --dis_layers=3 --lambda_g=5 --lambda_rec 2 --t_s_ratio=0.8 --save_latest_freq=10400 --batchSize 32 --gpu_id=0
```

### 4) Test the model

You can directly download our test results from Google Drive: **[Deepfashion](https://drive.google.com/drive/folders/1Y_Ar7w_CAYRgG2gzBg2vfxTCCen7q7k2?usp=sharing)**, **[Market1501](https://drive.google.com/drive/folders/15UBWEtGAqYaoEREIIeIuD-P4dRgsys19?usp=sharing)**.

**DeepFashion**
``` bash
python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion --batchSize 1 --gpu_id=0
```

**Market1501**

``` bash
python test.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --which_epoch latest --results_dir=./results/DPTN_market  --batchSize 1 --gpu_id=0
``` 

### 5) Evaluation

We adopt SSIM, PSNR, FID, LPIPS and person re-identification (re-id) system for the evaluation. Please clone the official repository **[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity/tree/future)** of the LPIPS score, and put the folder PerceptualSimilarity to the folder **[metrics](https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network/tree/main/metrics)**.

* For SSIM, PSNR, FID and LPIPS:

**DeepFashion**
``` bash
python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion --fid_real_path=./dataset/fashion/train --name=./fashion
``` 

**Market1501**

``` bash
python -m  metrics.metrics --gt_path=./dataset/market/test --distorated_path=./results/DPTN_market --fid_real_path=./dataset/market/train --name=./market --market
``` 

* For person re-id system:

Clone the code of the **[fast-reid](https://github.com/JDAI-CV/fast-reid)** to this project (`./fast-reid-master`). Move the **[config](https://drive.google.com/file/d/1xWCnNpcNrgjEMDKuK29Gre3sYEE1yWTV/view?usp=sharing)** and **[loader](https://drive.google.com/file/d/1axMKB7QlYQgo7f1ZWigTh3uLIDvXRxro/view?usp=sharing)** of the DeepFashion dataset to (`./fast-reid-master/configs/Fashion/bagtricks_R50.yml`) and (`./fast-reid-master/fastreid/data/datasets/fashion.py`) respectively. Download the **[pre-trained network](https://drive.google.com/file/d/1Co6NVWN6OSqPVUd7ut8xCwsQQDIOcypV/view?usp=sharing)** and put it under the `./fast-reid-master/logs/Fashion/bagtricks_R50-ibn/` directory. And then launch:

``` bash
python ./tools/train_net.py --config-file ./configs/Fashion/bagtricks_R50.yml --eval-only MODEL.WEIGHTS ./logs/Fashion/bagtricks_R50-ibn/model_final.pth MODEL.DEVICE "cuda:0"
``` 

### 6) Pre-trained Model

Our pre-trained models and logs can be downloaded from Google Drive: **[Deepfashion](https://drive.google.com/drive/folders/12Ufr8jkOwAIGVEamDedJy_ZWPvJZn8WG?usp=sharing)**[**[log](https://drive.google.com/drive/folders/16ZYYl_jVdK8E9FtnQi6oi6JGfBuD2jCt?usp=sharing)**], **[Market1501](https://drive.google.com/drive/folders/1YY_U2pMzLrZMTKoK8oBkMylR6KXnZJKP?usp=sharing)**[**[log](https://drive.google.com/drive/folders/1ujlvhz7JILULRVRJsLruT9ZAz2JCT74G?usp=sharing)**].

## Citation

```tex
@InProceedings{Zhang_2022_CVPR,
    author    = {Zhang, Pengze and Yang, Lingxiao and Lai, Jian-Huang and Xie, Xiaohua},
    title     = {Exploring Dual-Task Correlation for Pose Guided Person Image Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {7713-7722}
}
```
## Acknowledgement 

We build our project based on **[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)**. Some dataset preprocessing methods are derived from **[PATN](https://github.com/tengteng95/Pose-Transfer)**.

