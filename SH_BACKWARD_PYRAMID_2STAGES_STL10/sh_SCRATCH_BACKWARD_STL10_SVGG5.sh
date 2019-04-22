#!/bin/bash
#MOAB -l nodes=2:ppn=16:gpu

#MOAB -l walltime=2:00:00:00
#MSUB -N SVGG5_BACKWARD_PYRAMID_2STAGES_STL10
module load devel/cuda/8.0
module load lib/cudnn/5.1-cuda-8.0

cd code/SH_BACKWARD_PYRAMID_2STAGES_STL10

python main_2stages_BACKWARD_PYRAMID_2STAGES_STL10_SVGG5.py

