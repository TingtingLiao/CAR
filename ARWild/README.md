# MVP-Human Dataset for 3D Clothed Human Avatar Reconstruction from Multiple Frames
   
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2204.11184v2.pdf)

This repository contains the official PyTorch implementation of:

**MVP-Human Dataset for 3D Clothed Human Avatar Reconstruction from Multiple Frames**   
 
[//]: # ( ![]&#40;asset/overview.png&#41;)
  
 
  

## Installation    
  * **CUDA=10.2** 
  * Python = 3.7
  * PyTorch = 1.6.0 
  
**1. Setup virtual environment:**
```bash  
conda create -n arwild python=3.7
conda activate arwild

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
 

## Inference  
- Download the [pretrained models]( ) and put it in ./out/ckpt/ours-normal-1view/. 
- Download [extra data]( ) (PyMAF, ICON normal model, SMPL model) and put them to ./data.  
- Run the following script to test example images in directory ./examples. Results will be saved in ./examples/results.
``` 
python -m apps.infer --gpu 0 -cfg configs/arwild.yaml -in_dir "./examples/net"
```
 
 