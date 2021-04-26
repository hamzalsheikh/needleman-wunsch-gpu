#!/bin/bash

#SBATCH --job-name=nw

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rae66@aub.edu.lb

module load python/pytorch
module load cuda/latest

python gpudetection.py

make

./nw -0 -1 -2 -3

make clean
