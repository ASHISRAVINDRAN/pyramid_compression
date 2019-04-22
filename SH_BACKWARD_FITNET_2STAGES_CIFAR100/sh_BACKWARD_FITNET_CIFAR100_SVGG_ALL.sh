#!/bin/bash
#MOAB -l nodes=2:ppn=16:gpu

#MOAB -l walltime=2:00:00:00
#MSUB -N SVGG_ALL_BACKWARD_PYRAMID_2STAGES_STL10
#module load devel/cuda/8.0
#module load lib/cudnn/5.1-cuda-8.0

#cd code/SH_BACKWARD_FITNET_2STAGES_STL10
#python main_2stages_BACKWARD_FITNET_2STAGES_CIFAR100_SVGG17.py

python main_2stages_BACKWARD_FITNET_2STAGES_CIFAR100_SVGG14.py

python main_2stages_BACKWARD_FITNET_2STAGES_CIFAR100_SVGG11.py

python main_2stages_BACKWARD_FITNET_2STAGES_CIFAR100_SVGG8.py

python main_2stages_BACKWARD_FITNET_2STAGES_CIFAR100_SVGG5.py

