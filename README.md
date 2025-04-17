<div align="center">
<h1> FAM: Frequency-Based Adaptive Mutual Learning for Semi-Supervised Medical Image Segmentation </h1>
</div>

## 1. Installation
This repository is based on PyTorch 1.11.0, CUDA 11.3 and Python 3.8
```
conda create -n FAM python=3.8
conda activate FAM
pip install -r requirements.txt
```
## 2. Dataset
Data could be got at [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC) and [promise12](https://promise12.grand-challenge.org/Download/).
```
├── ./data
    ├── [ACDC]
        ├── [data]
        ├── test.list
        ├── train_slices.list
        ├── train.list
        └── val.list
    └── [promise12]
        ├── CaseXX_segmentation.mhd
        ├── CaseXX_segmentation.raw
        ├── CaseXX.mhd
        ├── CaseXX.raw
        ├── test.list
        └── val.list
```
## 3. Pretrained Backbone
Download pre-trained [Swin-Unet](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY) model to "./code/pretrained_ckpt" folder.
```
├── ./code/pretrained_ckpt
    └── swin_tiny_patch4_window7_224.pth
```
## 4. Usage
To train a model,
```
python ./code/train_ACDC.py  # for ACDC training
``` 
To test a model,
```
python ./code/test_ACDC.py  # for ACDC testing
```
## Acknowledgements
Our code is largely based on [ABD](https://raw.githubusercontent.com/chy-upc/ABD), [BCP](https://github.com/DeepMed-Lab-ECNU/BCP), and [SCP-Net](https://arxiv.org/pdf/2305.16214.pdf). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
