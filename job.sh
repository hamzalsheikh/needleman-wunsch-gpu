#!/bin/bash

#SBATCH --job-name=nw

#SBATCH --mem=12000
#SBATCH --partition=cudadev
#SBATCH --gres=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rae66@aub.edu.lb

module load cuda/latest

make

./nw -0 

make clean
