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
#os.getcwd()
print(os.getcwd())

from LoadDataSet import LoadDataSet
#from LoadModel import LoadModel
from LoadModel_gram_backward import LoadModel,Load_BigModel
from Load_Save_Model import save_model_stage1
from Load_Save_Model import save_model
from LoadModel import VGGNetNthLayer
from Train_2stages import Train_stage1
from Train_2stages import Train_stage2
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
#from torch.optim.lr_scheduler import MultiStepLR
#from torch.optim.lr_scheduler import StepLR

'''
POSSIBILITIES FOR MODEL NAME ARE
    'VGG19'
    'SVGG17_BACKWARD'
    'SVGG14_BACKWARD'
    'SVGG11_BACKWARD'
    'SVGG8_BACKWARD'
    'SVGG5_BACKWARD'

'''
model_name='SVGG11_BACKWARD'
print('model='+model_name)
#if model_name=='VGG19': #Is this if case necessary ?--Ashis
#    big_model_name='VGG19'




'''
Batch norm:
    BN=True performs Batch normalization
'''



'''
POSSIBLE DATASETS
    'CIFAR10'
    'CIFAR100'
    'STL10'
'''
dataset='STL10'

print('dataset='+dataset)

'''
POSSIBLE TRAIN MODES
    'PYRAMID' trains in the pyramid fashion
    'SCRATCH' trains from scratch using weights of VGG19
    'HINTON'  trains using idea of HINTON's paper (match last layer only)
    'WEIGHTS'
    'PYRAMID_2STAGES' trains in the pyramid fashion
    'PYRAMID_HINTON_2STAGES'
    'FITNET'
    'GRAM_PYRAMID'
'''
train_mode='PYRAMID_HINTON_2STAGES'
# Added new if case --Ashis

    
if train_mode=='WEIGHTS':
    if model_name[-3:]=='_bn':
        ensembleModel=VGGNetNthLayer(dataset,[13,26,39,52],model_name)
    else:
        ensembleModel=VGGNetNthLayer(dataset,[9,18,27,36],model_name)
        
print('Train mode='+train_mode)

Trainloader,Testloader=LoadDataSet(dataset).data
#model=LoadModel(model_name,dataset,train_mode).model
#print(model)
#save_model(model,[],[],[],[],[],model_name,train_mode,dataset,plot=False)

'''
model=LoadModel(model_name,dataset,train_mode).model
print('loading big model...')
big_model_name,ensembleModel=Load_BigModel(model_name,train_mode,dataset)    

if model_name=='VGG19':
    optimizer=torch.optim.Adam([{'params': model.features.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}], lr=1e-4, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
else:
    optimizer=torch.optim.Adam([{'params': model.features.parameters()},{'params': model.compressed_features.parameters()},
                            {'params': model.frozen_features.parameters(), 'lr': 1e-4},
                {'params': model.classifier.parameters(), 'lr': 1e-4}], lr=1e-4,
    betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
scheduler=ReduceLROnPlateau(optimizer, 'max',verbose=True,patience=5,eps=1e-8)
#scheduler = StepLR(optimizer, step_size=1, gamma=0.1)


trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters=Train(model,
                                                                                     optimizer,
                                                                                     Trainloader,
                                                                                     Testloader, 
                                                                                     epochs=10,
                                                                                     Train_mode=train_mode,
                                                                                     scheduler=scheduler,
                                                                                     big_model_name=big_model_name,
                                                                                     ensembleModel=ensembleModel)

save_model(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,
               model_name,train_mode,dataset,plot=False)
'''


#%%
model=LoadModel(model_name,dataset,train_mode).model
print('loading big model...')
big_model_name,ensembleModel=Load_BigModel(model_name,train_mode,dataset)

model = torch.nn.DataParallel(model) #make model DataParrallel
if (ensembleModel is not None):
    ensembleModel=torch.nn.DataParallel(ensembleModel)
    
print('model='+model_name)
print('big model='+big_model_name)
lr_stage1=1e-4
if 'fitnet'==train_mode.lower():
    optimizer=torch.optim.Adam([{'params': model.features.parameters()},{'params': model.compressed_features.parameters(), 'lr': lr_stage1*1},
                        {'params': model.frozen_features.parameters(), 'lr': lr_stage1*1},
                            {'params': model.regressor.parameters(), 'lr': lr_stage1*1},
                {'params': model.classifier.parameters(), 'lr': lr_stage1*1}], lr=lr_stage1,
    betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
else:
    optimizer=torch.optim.Adam(model.parameters(), lr=lr_stage1,
    betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
scheduler=ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=5,eps=1e-8,threshold=1e-20)
print ('STAGE1')
trainloss_to_file,testloss_to_file,Parameters=Train_stage1(model,optimizer,Trainloader,Testloader,epochs=None,Train_mode=train_mode,
                                                                                  Model_name=model_name,
                                                                                  Dataset=dataset,
                                                                                  scheduler=scheduler,
                                                                                  big_model_name=big_model_name,
                                                                                  ensembleModel=ensembleModel)
    
    #########################################################
    #####EVAL
   
trainAcc_to_file,testAcc_to_file,_,_,_=Train_stage2(model,optimizer,Trainloader,Testloader, epochs=0,
                                                                                                Train_mode=train_mode,
                                                                                                Model_name=model_name,
                                                                                                Dataset=dataset,
                                                                                                scheduler=scheduler,
                                                                                  big_model_name=big_model_name,
                                                                                  ensembleModel=ensembleModel)
    ##############################################################
save_model_stage1(model,trainloss_to_file,testloss_to_file,trainAcc_to_file,testAcc_to_file,Parameters, model_name,train_mode,dataset,plot=False)
#########    error-2
lr_stage2=1e-4#max(1e-5,10*optimizer.param_groups[-1]['lr'])
    
print ('STAGE2')
#    optimizer=torch.optim.Adam([{'params': model.features.parameters()},{'params': model.compressed_features.parameters()},
#                            {'params': model.frozen_features.parameters(), 'lr': lr_stage2*1},
#                {'params': model.classifier.parameters(), 'lr': lr_stage2*1}], lr=lr_stage2,betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
optimizer=torch.optim.Adam(model.parameters(), lr=lr_stage2, betas=(0.9, 0.999), eps=1e-08,weight_decay=0.0005)
scheduler=ReduceLROnPlateau(optimizer, 'max',verbose=True,patience=5,eps=1e-9,threshold=1e-20)
trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters=Train_stage2(model,optimizer,
                                                                                                Trainloader,Testloader,
                                                                                                epochs=None,
                                                                                                Train_mode=train_mode,
                                                                                                Model_name=model_name,
                                                                                                Dataset=dataset,
                                                                                                scheduler=scheduler,
                                                                                  big_model_name=big_model_name,
                                                                                  ensembleModel=ensembleModel)
    
    
    #error-2
save_model(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,
                   model_name,train_mode,dataset,plot=False)
