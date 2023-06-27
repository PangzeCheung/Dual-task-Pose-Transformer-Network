# Dual-task Pose Transformer Network
The source code for our paper "[
Pose Guided Person Image Generation via Dual-task Correlation and Affinity Learning](https://ieeexplore.ieee.org/abstract/document/10153661)“, Pengze Zhang, Lingxiao Yang, Jianhuang Lai, and Xiaohua Xie, TVCG 2023.
<img width="1148" alt="framework" src="https://user-images.githubusercontent.com/37894893/156797980-6387165c-3db8-48be-969f-011d3ecc3c05.png">

## Abstract

Pose Guided Person Image Generation (PGPIG) is the task of transforming a person's image from the source pose to a target pose. Existing PGPIG methods often tend to learn an end-to-end transformation between the source image and the target image, but do not seriously consider two issues: 1) the PGPIG is an ill-posed problem, and 2) the texture mapping requires effective supervision. In order to alleviate these two challenges, we propose a novel method by incorporating D ual-task P ose T ransformer N etwork and T exture A ffinity learning mechanism (DPTN-TA). To assist the ill-posed source-to-target task learning, DPTN-TA introduces an auxiliary task, i.e., source-to-source task, by a Siamese structure and further explores the dual-task correlation. Specifically, the correlation is built by the proposed Pose Transformer Module (PTM), which can adaptively capture the fine-grained mapping between sources and targets and can promote the source texture transmission to enhance the details of the generated images. Moreover, we propose a novel texture affinity loss to better supervise the learning of texture mapping. In this way, the network is able to learn complex spatial transformations effectively. Extensive experiments show that our DPTN-TA can produce perceptually realistic person images under significant pose changes. Furthermore, our DPTN-TA is not limited to processing human bodies but can be flexibly extended to view synthesis of other objects, i.e., faces and chairs, outperforming the state-of-the-arts in terms of both LPIPS and FID.


## Get Start

### 1) Train a model

**DeepFashion**
``` bash
python train.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --batchSize 32 --gpu_id=0
```
**Market1501**

``` bash
python train.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --dis_layers=3 --lambda_g=5 --lambda_rec 2 --t_s_ratio=0.8 --save_latest_freq=10400 --batchSize 32 --gpu_id=0
```

### 2) Test the model

You can directly download our test results from Google Drive: **[Deepfashion](https://drive.google.com/drive/folders/1Y_Ar7w_CAYRgG2gzBg2vfxTCCen7q7k2?usp=sharing)**, **[Market1501](https://drive.google.com/drive/folders/15UBWEtGAqYaoEREIIeIuD-P4dRgsys19?usp=sharing)**.

**DeepFashion**
``` bash
python test.py --name=DPTN_fashion --model=DPTN --dataset_mode=fashion --dataroot=./dataset/fashion --which_epoch latest --results_dir ./results/DPTN_fashion --batchSize 1 --gpu_id=0
```

**Market1501**

``` bash
python test.py --name=DPTN_market --model=DPTN --dataset_mode=market --dataroot=./dataset/market --which_epoch latest --results_dir=./results/DPTN_market  --batchSize 1 --gpu_id=0
```

## Citation

```tex
@ARTICLE{10153661,
  author={Zhang, Pengze and Yang, Lingxiao and Xie, Xiaohua and Lai, Jianhuang},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Pose Guided Person Image Generation via Dual-task Correlation and Affinity Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TVCG.2023.3286394}}
```
## Acknowledgement 

We build our project based on **[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)**. Some dataset preprocessing methods are derived from **[PATN](https://github.com/tengteng95/Pose-Transfer)**.

