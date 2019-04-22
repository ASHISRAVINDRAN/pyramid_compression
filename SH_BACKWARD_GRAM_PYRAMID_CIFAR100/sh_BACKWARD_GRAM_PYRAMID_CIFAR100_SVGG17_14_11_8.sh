#!/bin/bash
#MOAB -l nodes=1:ppn=16:gpu

#MOAB -l walltime=2:00:00:00
#MSUB -N SVGG8_BACKWARD_GRAM_PYRAMID_CIFAR100
module load devel/cuda/8.0
module load lib/cudnn/5.1-cuda-8.0

cd code/SH_BACKWARD_GRAM_PYRAMID_CIFAR100

python main_2stages_BACKWARD_GRAM_PYRAMID_CIFAR100_SVGG17.py
python main_2stages_BACKWARD_GRAM_PYRAMID_CIFAR100_SVGG14.py
python main_2stages_BACKWARD_GRAM_PYRAMID_CIFAR100_SVGG11.py

python main_2stages_BACKWARD_GRAM_PYRAMID_CIFAR100_SVGG8.py

