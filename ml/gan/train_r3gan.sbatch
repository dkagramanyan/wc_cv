#!/bin/bash
#SBATCH --gpus=4
#SBATCH --cpus-per-task=2
#SBATCH --time=04-0:0
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda_v11.2021

source activate torch

cd wc_cv/ml/gans/R3GAN/

srun python train.py \
        --outdir=./training-runs \
        --data=./dataset.zip \
        --gpus=1 \
        --batch=16 \
        --mirror=False \
        --aug=1 \
        --preset=FFHQ-256 \
        --tick=1 \
        --snap=200