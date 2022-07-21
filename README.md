# 3D Human Avatar Reconstruction from Unconstrained Frames

 
This repository contains the official PyTorch implementation of: **3D Human Avatar Reconstruction from Unconstrained Frames**  
 

## Installation 
**1. Setup virtual environment**

Go to the AR-Wild directory in the command line, then
```sh
$ sh ./install.sh
```
 
If you use other python and cuda versions (default python3.7 cuda 10.1), please change the cuda version and python version in ./install.sh

**2. Download the smpl model:** 

Download smpl models from https://smpl.is.tue.mpg.de/, put them into models folder under ./smpl/models/smpl
By default we use 10 PCA models and .pkl format.

## Evaluation  
**0. Generate 3D model from multiview images:**
```sh
$ conda activate ARWild
$ python -m apps.eval  --res_dir [your path]
```
**1. Render dancing avatar:**
```sh 
$ python -m apps.render_avatar --res_dir [your path]
```

## Trainning  
**0. Train Skinning Weight Network:**
```sh 
$ python -m apps.train_skin   
```

**1. Train Surfac Reconstruction Network:**
```sh 
$ python -m apps.train_surface
```

ensure that before training the skinning weight network, the surface reconstruction network 