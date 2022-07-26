# Clothed Avatar Reconstruction  
 
  
Clothed avatar reconstruction 
 
![Test Results](asset/image.png)
 
## Table of Contents 
- [Install](#install)
- [Dataset](#Dataset)  
- [Training](#Training)   
 
## Install   

## Dataset 
For single view reconstruction, please follow the [THuman2.0 Data Processing Instruction](docs/dataset.md) from ICON.
For avatar reconstruction, please follow the 

## Training 

#### Single image reconstruction 
Models are trained on THuman2.0 dataset using normal image as input and output human body in **projected
space**. Users can take RGB image as input by setting option '-ii rgb'. 
```bash
# PIFu  
python -m apps.train --gpu 0 --data thuman -ii normal -cfg configs/pifu.yaml  

# ICON (*: re-implementation)
python -m apps.train --gpu 0 --data thuman -ii normal -cfg configs/icon.yaml  

# Ours 
python -m apps.train --gpu 0 --data thuman -ii normal -cfg configs/pifu-sdf.yaml   
```

#### Avatar Reconstruction 
```bash  
# ARCH* (*: re-implementation)
python -m apps.train --gpu 0 --data mvp -ii normal -cfg configs/arch.yaml  

# ARCH++* (*: re-implementation)

# ARWild 
python -m apps.train --gpu 0 --data mvp -ii normal -cfg configs/arwild.yaml  

# Ours 
python -m apps.train --gpu 0 --data mvp -ii normal -cfg configs/ours.yaml  
```

## Testing  
1. Download pretrained models in  . 
2. Testing example images in directory ./examples. Results will be saved to ./examples/results.
``` 
python -m apps.infer --gpu 0 -cfg configs/ours.yaml  
```

## License