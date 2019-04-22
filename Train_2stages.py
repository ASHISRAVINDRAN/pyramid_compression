#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:25:09 2017

@author: enric
"""

import torch
import torch.nn as nn
import time
import sys
from Load_Save_Model import checkpoint_save
from Load_Save_Model import checkpoint_save_stage1

from torch.optim.lr_scheduler import ReduceLROnPlateau
loss_ce =nn.CrossEntropyLoss()
loss_mse= nn.MSELoss(size_average=False)
verborrea = False 
def Train_stage1(model,optimizer,TrainSet,TestSet,Train_mode,Model_name,Dataset,epochs=None,ensembleModel=None,weights=None,
          big_model_name=None,scheduler=None):
    if scheduler is None:
        scheduler=ReduceLROnPlateau(optimizer, 'min',verbose=True,patience=10,eps=1e-8)
    
    time_elapsed=0
    since_init=time.time()
    
    if 'pyramid' in Train_mode.lower():
        weights=None
    elif 'hinton_2stages' in Train_mode.lower():
        weights=None
    elif 'fitnet' in Train_mode.lower():
        weights=None
    elif Train_mode.lower()=='hinton':
        assert(len(weights)==2)
    elif Train_mode.lower()=='scratch':
        weights=None
    elif Train_mode.lower()=='weights':
        weights=weights
    else:
        print('Train mode error!')
        sys.exit()
        
    if 'hinton' in Train_mode.lower():
        STAGE_NUM=2
    else:
        STAGE_NUM=1
    try:
        model.module.stage=STAGE_NUM
        ensembleModel.module.stage=STAGE_NUM
    except:
        model.stage=STAGE_NUM
        ensembleModel.stage=STAGE_NUM
    max_lr,list_lr=update_list_lr(optimizer)
    trainloss_to_fil=[]
    testloss_to_fil=[]


    if isinstance(scheduler,ReduceLROnPlateau):
        patience_num=scheduler.patience
    else:
        if epochs==None:
            print('WARNING!!!!: Number of epochs not determined')
            sys.exit()
        patience_num='nothing'
        
    parameters=[[],[],patience_num,optimizer.param_groups[0]['weight_decay']]#first list for epochs, second for learning rate,3rd patience, 4th weight_decay,5 for time
    parameters[1].append(list_lr)
    
    epoch=0
    
    ####Temprorary code to load checkpoint --Ashis
    #model.load_state_dict(torch.load('CHECKPOINT/CHECKPOINT.t7'))  
    #print ('Checkpoint Loaded')
    if epochs==0:
        keep_training=False
    else:
        keep_training=True
        print ('INITIAL TEST STATISTICS')
        if 'pyramid' in Train_mode.lower():
            loss_test= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
        if 'hinton' in Train_mode.lower():
            loss_test= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
        elif 'fitnet' in Train_mode.lower(): 
            loss_test= train_fitnet_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
            
        checkpoint_save_stage1(model,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
        check_load=0
        
        if isinstance(scheduler,ReduceLROnPlateau):    
            scheduler.step(loss_test)
        else:
            best_test=loss_test
            scheduler.step()
        
    while keep_training:
        epoch=epoch+1
        if epochs !=None:
            if verborrea: 
                print('Epoch {}/{},  lr={}. patience={}, weight decay={}'.format(epoch, epochs,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))
        else:
            if verborrea:
                print('Epoch {}, lr={}, patience={}, weight decay={}'.format(epoch,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))

        if verborrea:
            print('-' * 10)
        if 'pyramid' in Train_mode.lower():
                #since=time.time()
                #train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                #time_elapsed=time_elapsed+(time.time()-since)
                
                if verborrea:
                    print ('TRAIN STATISTICS')
                train_loss= train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                if verborrea:
                    print ('TEST STATISTICS')
                test_loss= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
        elif 'hinton' in Train_mode.lower():
                #since=time.time()
                #train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                #time_elapsed=time_elapsed+(time.time()-since)
                
                if verborrea:
                    print ('TRAIN STATISTICS')
                train_loss= train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                if verborrea:
                    print ('TEST STATISTICS')
                test_loss= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
        elif 'fitnet' in Train_mode.lower(): 
                #train_fitnet_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                #time_elapsed=time_elapsed+(time.time()-since)
                
                if verborrea:
                    print ('TRAIN STATISTICS')
                train_loss= train_fitnet_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=False)
                if verborrea:
                    print ('TEST STATISTICS')
                test_loss= train_fitnet_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
                
        trainloss_to_fil.append(train_loss)
        testloss_to_fil.append(test_loss)

        
        if isinstance(scheduler,ReduceLROnPlateau):
            prev_num_bad_epochs=scheduler.num_bad_epochs
            if verborrea:
                print('-' * 10)
            save=(test_loss<scheduler.best)
            scheduler.step(test_loss)

            if save:
                checkpoint_save_stage1(model,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
                check_load=0
            if scheduler.num_bad_epochs==0 and prev_num_bad_epochs==scheduler.patience and not save:
                max_lr,list_lr=update_list_lr(optimizer)
                parameters[0].append(epoch)
                parameters[1].append(max_lr)
                model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))
                check_load=check_load+1
                if verborrea: print ('Checkpoint loaded')
    
            if max_lr<10*scheduler.eps or check_load==2:
                keep_training=False
        else:
            prev_max_lr=max_lr

            scheduler.step()
            max_lr,list_lr=update_list_lr(optimizer)
            if test_loss<=best_test:
                checkpoint_save_stage1(model,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
            if max_lr<prev_max_lr:
                parameters[0].append(epoch)
                parameters[1].append(max_lr)
                model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))            
                if verborrea:
                    print ('Checkpoint loaded')
            
        if epochs!=None:
            if epoch==epochs:
                keep_training=False
        
        if verborrea:
            print('-' * 10)
    if epochs!=0:
        model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))
        if verborrea:
            print ('Checkpoint loaded')

    parameters[0].append(epoch)
    if 'pyramid' in Train_mode.lower():
        print ('FINAL TRAIN STATISTICS')
        train_loss= train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=True)
        print ('FINAL TEST STATISTICS')
        test_loss= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
    elif 'hinton' in Train_mode.lower():
        print ('FINAL TRAIN STATISTICS')
        train_loss= train_pyramid_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=True)
        print ('FINAL TEST STATISTICS')
        test_loss= train_pyramid_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)
    elif 'fitnet' in Train_mode.lower():
        print ('FINAL TRAIN STATISTICS')
        train_loss= train_fitnet_stage1(ensembleModel,model,TrainSet,optimizer,big_model_name,eval=True)
        print ('FINAL TEST STATISTICS')
        test_loss= train_fitnet_stage1(ensembleModel,model,TestSet,optimizer,big_model_name,eval=True)


    trainloss_to_fil.append(train_loss)
    testloss_to_fil.append(test_loss)
    time_elapsed=time.time()-since_init
    print('Total time elapsed',time_elapsed)
    parameters.append(time_elapsed)
    return (trainloss_to_fil,testloss_to_fil,parameters)
##########################################################################################################
    
def Train_stage2(model,optimizer,TrainSet,TestSet,Train_mode,Model_name,Dataset,epochs=None,ensembleModel=None,weights=None,
          big_model_name=None,scheduler=None):
    if scheduler is None:
        scheduler=ReduceLROnPlateau(optimizer, 'max',verbose=True,patience=10,eps=1e-8)
    First_save=False

    time_elapsed=0
    try:
        model.module.stage=2
        if ensembleModel!=None:
            ensembleModel.module.stage=2    
    except:
        model.stage=2
        if ensembleModel!=None:
            ensembleModel.stage=2
    
    since_init=time.time()
    
    
    
    max_lr,list_lr=update_list_lr(optimizer)
    trainAcc_to_fil=[]
    testAcc_to_fil=[]
    trainloss_to_fil=[]
    testloss_to_fil=[]
    if isinstance(scheduler,ReduceLROnPlateau):
        patience_num=scheduler.patience
    else:
        if epochs==None:
            print('WARNING!!!!: Number of epochs not determined')
            sys.exit()
        patience_num='nothing'
        
    parameters=[[],[],patience_num,optimizer.param_groups[0]['weight_decay']]#first list for epochs, second for learning rate,3rd patience, 4th weight_decay,5 for time
    parameters[1].append(list_lr)
    
    epoch=0
    
    ####Temprorary code to load checkpoint --Ashis
    #model.load_state_dict(torch.load('CHECKPOINT/CHECKPOINT.t7'))  
    #print ('Checkpoint Loaded')
    if epochs==0:
        keep_training=False
    else:
        keep_training=True
        print ('INITIAL TEST STATISTICS')
    
        tofile_params2= train_scratch(model,TestSet,optimizer,eval=True)
       
            
        checkpoint_save(model,trainAcc_to_fil,testAcc_to_fil,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
        check_load=0
        
        if isinstance(scheduler,ReduceLROnPlateau):    
            scheduler.step(tofile_params2[0])
        else:
            best_test=tofile_params2[0]
            scheduler.step()
        
    while keep_training:
        epoch=epoch+1
        if epochs !=None:
            if verborrea:
                print('Epoch {}/{},  lr={}. patience={}, weight decay={}'.format(epoch, epochs,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))
        else:
            if verborrea: 
                print('Epoch {}, lr={}, patience={}, weight decay={}'.format(epoch,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))

        if verborrea: 
            print('-' * 10)

        #since=time.time()
        #train_scratch(model,TrainSet,optimizer,eval=False)
        #time_elapsed=time_elapsed+(time.time()-since)
                
        if verborrea: 
            print ('TRAIN STATISTICS')
        tofile_params1= train_scratch(model,TrainSet,optimizer,eval=False)
        if verborrea: 
            print ('TEST STATISTICS')
        tofile_params2= train_scratch(model,TestSet,optimizer,eval=True)
        
            
        trainAcc_to_fil.append(tofile_params1[0])
        testAcc_to_fil.append(tofile_params2[0])
        trainloss_to_fil.append(tofile_params1[1])
        testloss_to_fil.append(tofile_params2[1])
        
        if isinstance(scheduler,ReduceLROnPlateau):
            prev_num_bad_epochs=scheduler.num_bad_epochs
            if verborrea:
                print('-' * 10)
            
            save=(tofile_params2[0]>scheduler.best)
            scheduler.step(tofile_params2[0])
    
            if save:
                checkpoint_save(model,trainAcc_to_fil,testAcc_to_fil,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
                check_load=0
                First_save=True
            if scheduler.num_bad_epochs==0 and prev_num_bad_epochs==scheduler.patience and not save:
                max_lr,list_lr=update_list_lr(optimizer)
                parameters[0].append(epoch)
                parameters[1].append(max_lr)
                model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))
                check_load=check_load+1
                if verborrea:
                    print ('Checkpoint loaded')
    
            if max_lr<10*scheduler.eps or (check_load>=2 and First_save):
                keep_training=False
        else:
            prev_max_lr=max_lr

            scheduler.step()
            max_lr,list_lr=update_list_lr(optimizer)
            if tofile_params2[0]>=best_test:
                checkpoint_save(model,trainAcc_to_fil,testAcc_to_fil,trainloss_to_fil,testloss_to_fil,parameters,Model_name,Train_mode,Dataset)
            if max_lr<prev_max_lr:
                parameters[0].append(epoch)
                parameters[1].append(max_lr)
                model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))            
                if verborrea:
                    print ('Checkpoint loaded')
            
        if epochs!=None:
            if epoch==epochs:
                keep_training=False
        
        if verborrea:
            print('-' * 10)
    if epochs!=0:
        model.load_state_dict(torch.load('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset+'/CHECKPOINT.t7'))
        if verborrea:
            print ('Checkpoint loaded')

    parameters[0].append(epoch)
    print ('FINAL TRAIN STATISTICS')
    tofile_params1= train_scratch(model,TrainSet,optimizer,eval=True)
    print ('FINAL TEST STATISTICS')
    tofile_params2= train_scratch(model,TestSet,optimizer,eval=True)

    trainAcc_to_fil.append(tofile_params1[0])
    testAcc_to_fil.append(tofile_params2[0])
    trainloss_to_fil.append(tofile_params1[1])
    testloss_to_fil.append(tofile_params2[1])
    time_elapsed = time.time() - since_init
    print('Total time elapsed',time_elapsed)
    parameters.append(time_elapsed)
    return (trainAcc_to_fil,testAcc_to_fil,trainloss_to_fil,testloss_to_fil,parameters)




##########################################################################################################


            
            
def train_pyramid_stage1(ensembleModel,model,DataSet,optimizer,big_model_name=None,eval=True):
    _loss=0
#    _correct=0
#    _total=0
    if(eval):
       model.eval()
    else:
       model.train()
    ensembleModel.eval() 
    use_gpu = torch.cuda.is_available()       
    for batch_idx, (inputs, labels) in enumerate(DataSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
            ensembleModel.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
       
        outputs=  model(inputs)
        if isinstance(outputs,list):
            outputs=outputs[-1]
        
                
#        if not isinstance(outputs,list):
#            outputs=[outputs]
        ensemble_features= ensembleModel(inputs)
        if isinstance(ensemble_features,list):
            ensemble_features=ensemble_features[-1]
#        if big_model_name=='VGG19':
#            loss_output=[]       
#            loss_output.append(loss_mse(outputs[-1],ensemble_features))
#            loss_output.append(loss(outputs[-1], labels))
#            total_loss=sum(loss_output)       
#        else:
#            total_loss=loss_mse(outputs[1], ensemble_features[0])
        total_loss=loss_mse(outputs, ensemble_features)
        if(not eval):
            total_loss.backward()
            optimizer.step()
        
        _loss += total_loss.data[0]
#            _, predicted = torch.max(outputs[-1].data, 1)
#            _total += labels.size(0)
#            _correct += predicted.eq(labels.data).cpu().sum()
    
    
    _loss_average=_loss/len(DataSet.dataset)
#        _acc=_correct*100/_total
#        print('Accuracy: ',_acc)
    if verborrea:
        print('Loss: ',_loss)
    if verborrea:
        print('Average Loss: ',_loss_average)
        
    return _loss_average
    
    


def train_fitnet_stage1(ensembleModel,model,DataSet,optimizer,big_model_name=None,eval=True):
    _loss=0
#    _correct=0
#    _total=0
    if(eval):
       model.eval()
    else:
       model.train()
    ensembleModel.eval() 
    use_gpu = torch.cuda.is_available()       
    for batch_idx, (inputs, labels) in enumerate(DataSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
            ensembleModel.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
       
        outputs=  model(inputs)
#        if not isinstance(outputs,list):
#            outputs=[outputs]
        ensemble_features= ensembleModel(inputs)
#        if big_model_name=='VGG19':
#            loss_output=[]       
#            loss_output.append(loss_mse(outputs[-1],ensemble_features))
#            loss_output.append(loss(outputs[-1], labels))
#            total_loss=sum(loss_output)       
#        else:
#            total_loss=loss_mse(outputs[1], ensemble_features[0])
        total_loss=loss_mse(outputs, ensemble_features)
        if(not eval):
            total_loss.backward()
            optimizer.step()
        
        
        _loss += total_loss.data[0]
#            _, predicted = torch.max(outputs[-1].data, 1)
#            _total += labels.size(0)
#            _correct += predicted.eq(labels.data).cpu().sum()
    
    
    _loss_average=_loss/len(DataSet.dataset)
#        _acc=_correct*100/_total
#        print('Accuracy: ',_acc)
    if verborrea:
        print('Loss: ',_loss)
    if verborrea:
        print('Average Loss: ',_loss_average)
        
    return _loss_average

def train_weighted(ensembleModel,model,DataSet,optimizer,weights,eval=True):
    _loss=0
    _correct=0
    _total=0
    if(eval):
       model.eval()
    else:
       model.train()
    
    ensembleModel.eval()   
   
    use_gpu = torch.cuda.is_available()       
    for batch_idx, (inputs, labels) in enumerate(DataSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
       
        outputs=  model(inputs)
        if not isinstance(outputs,list):
            outputs=[outputs]
        loss_output=[]
        ensemble_features= ensembleModel(inputs)            
        for logit_s, logit_b,weight in zip(outputs,ensemble_features, weights):
            loss_output.append(weight*loss_mse(logit_s,logit_b))
        loss_output.append(weights[-1]*loss_ce(outputs[-1], labels))
        total_loss=sum(loss_output)                
        if(not eval):
            total_loss.backward()
            optimizer.step()
            
        _loss += total_loss.data[0]
        _, predicted = torch.max(outputs[-1].data, 1)
        _total += labels.size(0)
        _correct += predicted.eq(labels.data).cpu().sum()
    
    _loss_average=_loss/len(DataSet.dataset)
    _acc=_correct*100/_total
    if verborrea:
        print('Accuracy: ',_acc)
    if verborrea:
        print('Train Loss: ',_loss)
    if verborrea:
        print('Average Loss: ',_loss/len(DataSet.dataset))
        
    return [_acc,_loss_average]
    
def train_hinton(ensembleModel,model,DataSet,optimizer,weights,eval=True):
    _loss=0
    _correct=0
    _total=0
    if(eval):
       model.eval()
    else:
       model.train()
       
    ensembleModel.eval()   
       
    use_gpu = torch.cuda.is_available()       
    for batch_idx, (inputs, labels) in enumerate(DataSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
        
        outputs=  model(inputs)
        if not isinstance(outputs,list):
            outputs=[outputs]
        loss_output=[]
        ensemble_features= ensembleModel(inputs)
        for logit_s, logit_b,weight in zip(outputs[-2:],ensemble_features[-2:], weights):
            loss_output.append(weight*loss_mse(logit_s,logit_b))
        loss_output.append(weights[-1]*loss_ce(outputs[-1], labels))
    
        total_loss=sum(loss_output)             
        if(not eval):
            total_loss.backward()
            optimizer.step()
        _loss += total_loss.data[0]
        _, predicted = torch.max(outputs[-1].data, 1)
        _total += labels.size(0)
        _correct += predicted.eq(labels.data).cpu().sum()
    
    _loss_average=_loss/len(DataSet.dataset)
    _acc=_correct*100/_total
    if verborrea: 
        print('Accuracy: ',_acc)
    if verborrea: 
        print('Loss: ',_loss)
    if verborrea: 
        print('Average Loss: ',_loss/len(DataSet.dataset))
        
    return [_acc,_loss_average]
   

def train_scratch(model,DataSet,optimizer,eval=True):
    
    _loss=0
    _correct=0
    _total=0
    if(eval):
       model.eval()
    else:
       model.train()
       
    use_gpu = torch.cuda.is_available()       
    for batch_idx, (inputs, labels) in enumerate(DataSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        optimizer.zero_grad()
       
        outputs=  model(inputs)
        if not isinstance(outputs,list):
            outputs=[outputs]
        total_loss=loss_ce(outputs[-1], labels)
        if(not eval):
            total_loss.backward()
            optimizer.step()
        
        _loss += total_loss.data[0]
        _, predicted = torch.max(outputs[-1].data, 1)
        _total += labels.size(0)
        _correct += predicted.eq(labels.data).cpu().sum()
    
    _loss_average=_loss/len(DataSet.dataset)
    _acc=_correct*100/_total
    if verborrea: 
        print('Accuracy: ',_acc)
    if verborrea:
        print('Loss: ',_loss)
    if verborrea:
        print('Average Loss: ',_loss/len(DataSet.dataset))
    
    return [_acc,_loss_average]   


def update_list_lr(optimizer):
    list_lr=[]
    for param in optimizer.param_groups:
        list_lr.append(param['lr'])
    max_lr=max(list_lr)
    return max_lr,list_lr
'''
def eval_model(model,TrainSet, Train_mode,big_model_name,weights=None,TestSet=None,ensembleModel=None):
    use_gpu = torch.cuda.is_available()
    train_loss=0
    train_correct=0
    train_total=0
    test_loss=0
    test_correct=0
    test_total=0
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(TrainSet):
        if use_gpu:
            inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            model.cuda()
        else:
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        
        outputs=  model(inputs)
        if big_model_name=='VGG19':
            outputs=[outputs]
        total_loss=loss(outputs[-1], labels)  
            
#        if Train_mode.lower()=='pyramid':
#            total_loss =loss_pyramid(ensembleModel,outputs,inputs,labels,big_model_name)
#        elif Train_mode.lower()=='hinton':
#            total_loss =loss_hinton(ensembleModel,outputs,inputs,labels,weights)
#        elif Train_mode.lower()=='scratch':
#            total_loss =loss_scratch(outputs,labels,big_model_name)
#        elif Train_mode.lower()=='weights':
#            total_loss =loss_weighted(ensembleModel,outputs,inputs,labels,weights)
#        
           
        train_loss += total_loss.data[0]
        #train_loss += total_loss
        _, predicted = torch.max(outputs[-1].data, 1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels.data).cpu().sum()
        
    train_loss_average=train_loss/len(TrainSet.dataset)
    train_acc=train_correct*100/train_total
    #print('\nEpoch: '+str(epoch)+',Train Accuracy: '+str(train_acc)+',Train Loss='+str(train_loss))
    print('Train Accuracy: ',train_acc)
    print('Train Loss: ',train_loss)
    print('Average Train Loss: ',train_loss/len(TrainSet.dataset))
    print('\n')
    if TestSet!=None:
        for batch_idx, (inputs, labels) in enumerate(TestSet):
            if use_gpu:
                inputs, labels = torch.autograd.Variable(inputs.cuda()),torch.autograd.Variable(labels.cuda())
            else:
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            
            outputs_test = model(inputs)
            if big_model_name=='VGG19':
                outputs_test=[outputs_test]
            #_, preds = torch.max(outputs_test.data, 1)
            if Train_mode.lower()=='pyramid':
                total_loss =loss_pyramid(ensembleModel,outputs_test,inputs,labels,big_model_name)
            elif Train_mode.lower()=='hinton':
                total_loss =loss_hinton(ensembleModel,outputs_test,inputs,labels,weights)
            elif Train_mode.lower()=='scratch':
                total_loss =loss_scratch(outputs_test,labels,big_model_name)
            elif Train_mode.lower()=='weights':
                total_loss =loss_weighted(ensembleModel,outputs_test,inputs,labels,weights)      
        
            # statistics
            test_loss += total_loss.data[0]
            _, predicted = torch.max(outputs_test[-1].data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels.data).cpu().sum()
            
        test_loss_average=test_loss/len(TestSet.dataset)
        test_acc=test_correct*100/test_total
        print('Test Accuracy: ',test_acc)
        print('Test Loss: ',test_loss)
        print('Average Test Loss: ',test_loss/len(TestSet.dataset))
        print('\n')    
        

    
    return [train_acc,test_acc,train_loss_average,test_loss_average]
'''



            
