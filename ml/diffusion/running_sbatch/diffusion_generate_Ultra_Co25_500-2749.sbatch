#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4-0:0
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda_v11.2021

source activate torch

BASE_DIR="/home/dgkagramanyan/wc_cv/ml/diffusion/runs"

MODEL_BASE="256-768/o_bc_left_4x_768_360_768_median_Ultra_Co25"

OUTPUT_SUBDIR="256-768/o_bc_left_4x_768_360_768_median_Ultra_Co25/generated"

START=500
BATCH=10
EPOCHS=225

srun python diffusion_generate.py \
    --model_path "${BASE_DIR}/${MODEL_BASE}/0099.pth" \
    --test_dir "${BASE_DIR}/${OUTPUT_SUBDIR}" \
    --start_step $START \
    --batch_size $BATCH \
    --epochs $EPOCHS