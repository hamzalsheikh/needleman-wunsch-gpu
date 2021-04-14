#!/bin/bash

#SBATCH --job-name=nw

#SBATCH --partition=cudadev
#SBATCH --gres=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rae66@aub.edu.lb

module load python/3
module load python/pytorch
module load cuda/latest

python gpudetection.py

make

./nw -N 10 -0 -1
./nw -N 50 -0 -1
./nw -N 100 -0 -1
./nw -N 500 -0 -1
./nw -N 1000 -0 -1
./nw -N 2500 -0 -1
./nw -N 5000 -0 -1
./nw -N 7500 -0 -1
./nw -N 10000 -0 -1
./nw -N 12500 -0 -1
./nw -N 15000 -0 -1
./nw -N 17500 -0 -1
./nw -N 20000 -0 -1
./nw -N 24000 -0 -1
./nw -N 28000 -0 -1
./nw -0 -1

make clean
