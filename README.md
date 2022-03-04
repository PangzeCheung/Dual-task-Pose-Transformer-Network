# Dual-task Pose Transformer Network
The source code for our paper "Exploring Dual-task Correlation for Pose Guided Person Image Generation“ (CVPR2022)

<img width="1148" alt="framework" src="https://user-images.githubusercontent.com/37894893/156797980-6387165c-3db8-48be-969f-011d3ecc3c05.png">

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


* Download the DeepFashion dataset **[in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)**, and put them under the `./dataset/fashion` directory.

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
python train.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --dis_layer=3 --lambda_g=5 --lambda_rec 2 --t_s_ratio=0.8 --save_latest_freq=10400 --batchSize 32 --gpu_id=0
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

We adopt SSIM, PSNR, FID and LPIPS for the evaluation.

**DeepFashion**
``` bash
python -m  metrics.metrics --gt_path=./dataset/fashion/test --distorated_path=./results/DPTN_fashion --fid_real_path=./dataset/fashion/train --name=./fashion
``` 

**Market1501**

``` bash
python -m  metrics.metrics --gt_path=./dataset/market/test --distorated_path=./results/DPTN_market --fid_real_path=./dataset/market/train --name=./market --market
``` 


### 6) Pre-trained Model

Our pre-trained model can be downloaded from Google Drive: **[Deepfashion](https://drive.google.com/drive/folders/12Ufr8jkOwAIGVEamDedJy_ZWPvJZn8WG?usp=sharing)**, **[Market1501](https://drive.google.com/drive/folders/1YY_U2pMzLrZMTKoK8oBkMylR6KXnZJKP?usp=sharing)**.

## Citation

```tex

```
## Acknowledgement 

We build our project based on **[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)**. Some dataset preprocessing methods are derived from **[PATN](https://github.com/tengteng95/Pose-Transfer)**.

