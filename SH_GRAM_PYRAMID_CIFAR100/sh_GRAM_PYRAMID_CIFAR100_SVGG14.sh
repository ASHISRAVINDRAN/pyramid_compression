#!/bin/bash
#MOAB -l nodes=2:ppn=16:gpu

#MOAB -l walltime=2:00:00:00
#MSUB -N SVGG14_GRAM_PYRAMID_CIFAR100
module load devel/cuda/8.0
module load lib/cudnn/5.1-cuda-8.0

cd code/SH_GRAM_PYRAMID_CIFAR100

python main_2stages_GRAM_PYRAMID_CIFAR100_SVGG14.py



