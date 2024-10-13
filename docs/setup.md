# Environment Setup

## Base Environment 
Python == 3.8  (>=3.8) \
CUDA == 11.8  (>=11.2) \
PyTorch == 2.0.1  (>=1.10.0) \
mmdet3d == 1.0.0rc6

## Step-by-step installation instructions

Following [mmdetection3D](https://github.com/open-mmlab/mmdetection3d)


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n viewformer python=3.8 -y
conda activate viewformer
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

**e. Install mmdet3d.**
```shell
pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd ./ViewFormer-Occ
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```

**f. Compile CUDA operators of DCNv3 (optional: use the InternImage backbone only).**
```shell
cd ./ops_dcnv3
sh ./make.sh
```
