#!/bin/bash
#MOAB -l nodes=1:ppn=16:gpu

#MOAB -l walltime=2:00:00:00
#MSUB -N SVGG5_BACKWARD_PYRAMID_2STAGES_CIFAR100


#module load devel/cuda/8.0
#module load lib/cudnn/5.1-cuda-8.0

#cd code/SH_HINTON_2STAGES_STL10


python main_HINTON_2STAGES_STL10_SVGG17.py

python main_HINTON_2STAGES_STL10_SVGG14.py

python main_HINTON_2STAGES_STL10_SVGG11.py

python main_HINTON_2STAGES_STL10_SVGG8.py


python main_HINTON_2STAGES_STL10_SVGG5.py