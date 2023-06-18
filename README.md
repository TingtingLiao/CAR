# High-Fidelity Clothed Avatar Reconstruction from a Single Image
   
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/submit/4833190/view)

This repository contains the official PyTorch implementation of:

**High-Fidelity Clothed Avatar Reconstruction from a Single Image**   
 
 ![](asset/overview.png)
  
 
## Table of Contents 
- [Installation](#Installation)
- [Dataset](#Dataset)  
- [Training](#Training)   
- [Inference](#Inference)   

## Installation    
  * **CUDA=10.2** 
  * Python = 3.7
  * PyTorch = 1.6.0 
  
**1. Setup virtual environment:**
```bash  
conda create -n car python=3.7
conda activate car

# install pytorch
conda install -c pytorch pytorch=1.10.0 torchvision==0.7.0 cudatoolkit=10.2

# install pytorch3d 
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt1100/download.html

# install trimesh  
conda install -c conda-forge rtree pyembree
pip install trimesh[all]

# install other dependencies
pip install -r requirement.txt

# install customized smpl code
cd smpl
python setup.py install
cd ../
```
If you use other python and cuda versions (default python3.7 cuda 10.2), please change the cuda version and python version in ./install.sh. If you use other pytorch version (default pytorch 1.6.0), please install pytorch3d according to the official install instruction official [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).  

 
**2. Download smpl models from https://smpl.is.tue.mpg.de/, put them into models folder under ./data/smpl_related/models/smpl/**

  
[//]: # (## Dataset  )

[//]: # (**coming soon.**)

## Training  
```bash  
# CAR 
python -m apps.train -cfg configs/car-rp.yaml --gpu 0 

# ARCH* (*: re-implementation)
python -m apps.train -cfg configs/arch.yaml --gpu 0  
```
The results will be saved in ./out/. 

## Inference  
- Download the [pretrained models]() and put it in ./out/ckpt/ours-normal-1view/. 
- Download [extra data]() (PyMAF, ICON normal model, SMPL model) and put them to ./data.  
- Run the following script to test example images in directory ./examples. Results will be saved in ./examples/results.
```
python -m apps.infer --gpu 0 -cfg configs/car-rp.yaml

```



## Citation

```latex
@inproceedings{liao2023car,
  title     = {{High-Fidelity Clothed Avatar Reconstruction from a Single Image}},
  author    = {Liao, Tingting and Zhang, Xiaomei and Xiu, Yuliang and Yi, Hongwei and Liu, Xudong and Qi, Guo-Jun and Zhang, Yong and Wang, Xuan and Zhu, Xiangyu and Lei, Zhen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
}
```