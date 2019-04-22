# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:03:04 2017

@author: enric
"""

import sys
import os
sys.path.append(os.getcwd())
os.chdir("..")
sys.path.append(os.getcwd())
print(os.getcwd())


from LoadDataSet import LoadDataSet
from LoadModel import LoadModel,Load_BigModel
from Load_Save_Model import save_model
from LoadModel import VGGNetNthLayer
from Train import Train
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

'''
POSSIBILITIES FOR MODEL NAME ARE
    'VGG19'
    'SVGG17'
    'SVGG14'
    'SVGG11'
    'SVGG8'
    'SVGG5'

'''
model_name='VGG19'
print('model='+model_name)

'''
POSSIBLE DATASETS
    'CIFAR10'
    'CIFAR100'
    'STL10'
    'SVHN'
'''
dataset='CIFAR100'

print('dataset='+dataset)

'''
POSSIBLE TRAIN MODES
    'PYRAMID' trains in the pyramid fashion
    'PYRAMID_2loss' trains in the pyramid fashion with 2 losses
    'SCRATCH' trains from scratch using weights of VGG19
    'HINTON'  trains using idea of HINTON's paper (match last layer only)
    'WEIGHTS'
'''
train_mode='SCRATCH'
if train_mode=='SCRATCH':
    big_model_name =None
    
if train_mode=='WEIGHTS':
    if model_name[-3:]=='_bn':
        ensembleModel=VGGNetNthLayer(dataset,[13,26,39,52],model_name)
    else:
        ensembleModel=VGGNetNthLayer(dataset,[9,18,27,36],model_name)
        
print('Train mode='+train_mode)

Trainloader,Testloader=LoadDataSet(dataset).data
big_model_name,ensemblModel=Load_BigModel(model_name,train_mode,dataset)
model=LoadModel(model_name,dataset,train_mode).model
optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)    
scheduler=ReduceLROnPlateau(optimizer, 'max',verbose=True,patience=5,eps=1e-9)
#%%
trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters=Train(model,
                                                                                         optimizer,
                                                                                         Trainloader,
                                                                                         Testloader,
                                                                                         Model_name=model_name,
                                                                                         dataset= dataset,
                                                                                         epochs=None,
                                                                                         Train_mode=train_mode,
                                                                                         scheduler=scheduler,
                                                                                         big_model_name=big_model_name,
                                                                                         ensembleModel=ensemblModel)
    
save_model(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,
                   model_name,train_mode,dataset,plot=False)
