# Dual-task Pose Transformer Network
The source code for our paper "[Exploring Dual-task Correlation for Pose Guided Person Image Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Exploring_Dual-Task_Correlation_for_Pose_Guided_Person_Image_Generation_CVPR_2022_paper.pdf)“, Pengze Zhang, Lingxiao Yang, Jianhuang Lai, and Xiaohua Xie, CVPR 2022. Video: [[Chinese](https://www.koushare.com/video/videodetail/35887)] [[English](https://www.youtube.com/watch?v=p9o3lOlZBSE)]
<img width="1148" alt="framework" src="https://user-images.githubusercontent.com/37894893/156797980-6387165c-3db8-48be-969f-011d3ecc3c05.png">

## Abstract

Pose Guided Person Image Generation (PGPIG) is the task of transforming a person image from the source pose to a given target pose. Most of the existing methods only focus on the ill-posed source-to-target task and fail to capture reasonable texture mapping. To address this problem, we propose a novel Dual-task Pose Transformer Network (DPTN), which introduces an auxiliary task (i.e., source-tosource task) and exploits the dual-task correlation to promote the performance of PGPIG. The DPTN is of a Siamese structure, containing a source-to-source self-reconstruction branch, and a transformation branch for source-to-target generation. By sharing partial weights between them, the knowledge learned by the source-to-source task can effectively assist the source-to-target learning. Furthermore, we bridge the two branches with a proposed Pose Transformer Module (PTM) to adaptively explore the correlation between features from dual tasks. Such correlation can establish the fine-grained mapping of all the pixels between the sources and the targets, and promote the source texture transmission to enhance the details of the generated target images. Extensive experiments show that our DPTN outperforms state-of-the-arts in terms of both PSNR and LPIPS. In addition, our DPTN only contains 9.79 million parameters, which is significantly smaller than other approaches.


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

