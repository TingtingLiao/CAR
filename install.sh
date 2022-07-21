#/bin/bash
echo "Set up enviorment for AR-Wild..."

conda create -n ARWild python=3.7
conda activate ARWild

echo "Installing pytorch..."
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=10.2

echo "Installing pytorch3d..."
echo "If you are using other versions (default python3.7 cuda 10.2), change the cuda version and python version in ./install.sh"
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt1100/download.html

echo "Installing trimesh..."
conda install -c conda-forge rtree pyembree
pip install trimesh[all]

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Installing customized smpl code"
cd smpl
python setup.py install
cd ../

# optional for arch++
# pip install pointnet2_ops_lib/.
# pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

echo "Done!"