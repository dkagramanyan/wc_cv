wc-cv
=====
[![DOI](https://zenodo.org/badge/423016150.svg)](https://zenodo.org/badge/latestdoi/423016150)

Tools for analisys of WC-Co's microstructure SEM images. 

Series of abstracs:

1) [Computer analysis of the cemented carbides’ microstructure](https://lettersonmaterials.com/en/Readers/Article.aspx?aid=41463)
2) [Topology of WC/Co Interfaces in Cemented Carbides](https://doi.org/10.3390/ma16165560) 

 [Documentation](https://dkagramanyan.github.io/wc_cv/)


Torch env
1) Python/Anaconda
2) CUDA/12.4
3) gnu9/9.3
4) python 3.13

torch install
linux+cuda12.6+pip3 install torch torchvision

R3GAN

2) pip install click Ninja tensorboard
3) conda install -c conda-forge libstdcxx-ng

betas .0

```
python dataset_tool.py --source=./data/o_bc_left_4x_768_360_median_Ultra_Co11_rgb --dest=./datasets/o_bc_left_4x_768_360_median_Ultra_Co11_rgb_768x768.zip     
```

```
python dataset_tool.py --source=./data/o_bc_left_4x_768_360_median_Ultra_Co11_rgb --dest=./datasets/o_bc_left_4x_768_360_median_Ultra_Co11_rgb_512x512.zip --resolution=512x512
```

```
python dataset_tool.py --source=./data/o_bc_left_4x_768_360_median_Ultra_Co11_rgb --dest=./datasets/o_bc_left_4x_768_360_median_Ultra_Co11_rgb_256x256.zip --resolution=256x256
```

training, 24gb vram

```
python train.py --outdir=./training-runs --data=./datasets/o_bc_left_4x_768_360_median_Ultra_Co11_rgb_256x256.zip --gpus=1 --batch=24 --mirror=1 --aug=1 --preset=FFHQ-256 --tick=1 --snap=100
```
