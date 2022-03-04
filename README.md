# DPTN
The source code for our paper "Exploring Dual-task Correlation for Pose Guided Person Image Generation“ (CVPR2022)

<img width="1148" alt="framework" src="https://user-images.githubusercontent.com/37894893/156797980-6387165c-3db8-48be-969f-011d3ecc3c05.png">

## Get Start

### 1) Installation

**Requirements**

* Python 3
* pytorch (1.0.0)
* CUDA
* visdom

**Conda installation**

```bash
# 1. Create a conda virtual environment.
conda create -n gfla python=3.6 -y
source activate gfla

# 2. Install dependency
pip install -r requirement.txt

# 3. Build pytorch Custom CUDA Extensions
./setup.sh
```

### 2) Data Preperation

Following **[PATN](https://github.com/tengteng95/Pose-Transfer)**, the dataset split files and extracted keypoints files can be obtained as follows:

**DeepFashion**


* 1. Download the DeepFashion dataset **[in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)**, and put them under the `./dataset/fashion` directory.
* 2. Split the raw image into the training set (`./dataset/fashion/train`) and test set (`./dataset/fashion/test`):
``` bash
python script/generate_fashion_datasets.py
```
* 3. Download train/test pairs and train/test keypoints annotations from Google Drive or Baidu Disk, including **fasion-resize-pairs-train.csv, fasion-resize-pairs-test.csv, fasion-resize-annotation-train.csv, fasion-resize-annotation-train.csv**, and put them under the `./dataset/fashion` directory.

**Market1501**

* 1. Download the Market1501 dataset from **[here](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)**. Rename **bounding_box_train** and **bounding_box_test** as train and test, and put them under the `./dataset/market` directory.

* 2. Download train/test key points annotations from Google Drive including **market-pairs-train.csv, market-pairs-test.csv, market-annotation-train.csv, market-annotation-train.csv**. Put these files under the `./dataset/market` directory.

### 3) Train a model

**DeepFashion**
``` bash
python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 32 --gpu_id=0
```
**Market1501**

``` bash
python3 train.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --dis_layer=3 --lambda_g=5 --lambda_rec 2 --t_s_ratio=0.8 --save_latest_freq=10400 --batchSize 32 --gpu_id=0
```

### 4) Test the model

**DeepFashion**
``` bash
python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion --batchSize 32 --gpu_id=0
```

**Market1501**

``` bash
python3 test.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --which_epoch latest --results_dir=./results/DPTN_market  --batchSize 32 --gpu_id=0
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

Our pre-trained model can be downloaded from Google Drive.

## Citation

```tex

```
## Acknowledgement 

We build our project based on (https://github.com/RenYurui/Global-Flow-Local-Attention & https://github.com/daa233/generative-inpainting-pytorch). Some dataset preprocessing methods are derived from (https://github.com/tengteng95/Pose-Transfer).

