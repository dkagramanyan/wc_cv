#!/bin/bash
#SBATCH --job-name=autoencoder_train
#SBATCH --gpus=1
#SBATCH --cpus-per-task 6
#SBATCH --constraint="type_c"

module load Python/Anaconda_v11.2021

# ��������� ��������� Anaconda
# ����� ���������� ��������� ������������� �������������� �������
conda deactivate
source activate tf2

echo "Working on node `hostname`"
echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
srun python vae_train.py
