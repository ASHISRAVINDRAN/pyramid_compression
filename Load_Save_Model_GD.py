# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:55:17 2017

@author: enric
"""
import torch
import os.path
import shutil
verborrea = True 

def load_model(model,Model_name,Train_mode,Dataset):
    #Added upper() for name consistency --Ashis
    Model_name=Model_name.upper()
    Train_mode=Train_mode.upper()
    Dataset =Dataset.upper()
    #############################
    print('File to be loaded:'+Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7')
    if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'):
            try:
                model=model.module
            except:
                pass 
            print('Loading File: '+Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7')
            model.load_state_dict(torch.load(Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'))
            return model
    else:
            print ('WARNING!!!: Weight of '+Model_name+' not loaded. No Existing file')
            return model
####################################################################################################################3








def save_model(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,
               Model_name,Train_mode,Dataset,plot=False):
        try:
            model=model.module
        except:
            pass 
        Model_name=Model_name.upper()
        Train_mode=Train_mode.upper()
        Dataset =Dataset.upper()
        path=Dataset+'/'+Train_mode+'/'+Model_name+'/'
        if not os.path.exists(path):
            os.makedirs(path)
            
        if "_2STAGES" in Train_mode:
            stage="STAGE2_"
        else:
            #stage=None
            stage=''
        torch.save(model.state_dict(),Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7')
        print(Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'+' saved')  
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Testacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in testAcc_to_file:
            thefile.write("%s," % item)
        thefile.close()
    
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainacc_'+stage+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in trainAcc_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'w')
        thefile.write('%s \n' %stage)
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_grooups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
        thefile.write("\n\n" )

        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum of the learning rates decreased to %s \n" %(epoch, lr))
        thefile.write("Trained for %s epochs \n\n" %Parameters[0][-1])
        
        thefile.write("Train Statistics \n")
        thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
        thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
        thefile.write("Test Statistics \n")
        thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
        thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
        for i in range(len(testAcc_to_file)):
            if testAcc_to_file[i]==testAcc_to_file[-1]:
                break
        if i+1==len(testAcc_to_file):
            i=-1
        thefile.write('Maximum test accuracy in epoch %s (if 0  it means that the initial state was the best)\n\n'%str(i+1))
        
        thefile.write('Total time elapsed %s\n\n' %Parameters[4])

        thefile.write(20*'-'+'\n\n')


        thefile.close() 
        shutil.rmtree('CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset)



###############################################################################################################
def save_model_stage1(model,trainloss_to_file,testloss_to_file,trainAcc_to_file,testAcc_to_file,parameters,Model_name,Train_mode,Dataset,plot=False):
        try:
            model=model.module
        except:
            pass 
        
        Model_name=Model_name.upper()
        Train_mode=Train_mode.upper()
        Dataset =Dataset.upper()
        path=Dataset+'/'+Train_mode+'/'+Model_name+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(),Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7')
        print(Dataset+'/'+Train_mode+'/'+Model_name+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'+' saved')  
        

        for i in range(0,len(trainloss_to_file)):
            if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv'):
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv', 'a')
            else:
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv', 'w')
            
            for item in trainloss_to_file[i]:
                thefile.write("%s," % item)
            thefile.close() 
        
        
            if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv'):
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv', 'a')
            else:
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'_'+str(5-i)+'.csv', 'w')
            for item in testloss_to_file[i]:
                thefile.write("%s," % item)
            thefile.close() 
        
        
        opt_num=0    
        for Parameters in parameters:
            if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt'):
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'a')
            else:
                thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'w')
        
            if 'BACKWARD' in Model_name.upper():
                fnum=opt_num+2
            else:
                fnum=5-opt_num
            
            
            thefile.write("STAGE1 Features"+str(fnum)+"\n")
            thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))
            
            if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
                for i in range(len(Parameters[1][0])):
                    thefile.write("Initial learning rate for param_grooups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
            else:
                thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
            thefile.write("\n\n" )
    
            for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
                thefile.write("In epoch %s, maximum of the learning rates decreased to %s \n" %(epoch, lr))
            thefile.write("Trained for %s epochs \n\n" %Parameters[0][-1])
            
            thefile.write("Train Statistics \n")
            thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
            thefile.write('MSE Average Loss: %s \n\n'%trainloss_to_file[opt_num][-1])
            
            thefile.write("Test Statistics \n")
            thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
            thefile.write('MSE Average Loss: %s \n\n'%testloss_to_file[opt_num][-1])
            opt_num=opt_num+1
    #        for i in range(len(trainloss_to_file)):
    #            if trainloss_to_file[i]==trainloss_to_file[-1]:
    #                break
    #        if i+1==len(trainloss_to_file):
    #            i=-1
    #        thefile.write('Minimum loss accuracy in epoch %s (if 0  it means that the initial state was the best)\n\n'%str(i+1))
    #        
            thefile.write('Total time elapsed %s\n\n' %Parameters[4])
    
            thefile.write(20*'-'+'\n\n')
    
    
            thefile.close() 
        
        
        #SAVE IN FOLDER STAGE1
        
        folder='STAGE1'
       
        src =Dataset+'/'+Train_mode+'/'+Model_name+'/'
        if not os.path.exists(Dataset+'/'+Train_mode+'/'+Model_name+'/'+folder):
            os.makedirs(Dataset+'/'+Train_mode+'/'+Model_name+'/'+folder)
        src_files = os.listdir(src)
        dest = src+folder
        
        for file_name in src_files:
            if file_name!='STAGE1':
                full_file_name = os.path.join(src, file_name)
                if (os.path.isfile(full_file_name)):
                    shutil.copy(full_file_name, dest)
        '''       
        if not os.path.exists(Dataset+'/'+Train_mode+'/'+Model_name+folder):
            os.makedirs(Dataset+'/'+Train_mode+'/'+Model_name+folder)
        torch.save(model.state_dict(),Dataset+'/'+Train_mode+'/'+Model_name+folder+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7')
        print(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/'+Model_name+'_'+Train_mode+'_'+Dataset+'.t7'+' saved')  
        

        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Trainloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Testloss_STAGE1_'+Model_name+'_'+Train_mode+'_'+Dataset+'.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        '''
        
'''        
        if os.path.isfile(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt'):
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'a')
        else:
            thefile = open(Dataset+'/'+Train_mode+'/'+Model_name+folder+'/Parameters_'+Model_name+'_'+Train_mode+'_'+Dataset+'.txt', 'w')

        thefile.write("STAGE1 \n")
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_grooups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
        thefile.write("\n\n" )

        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum of the learning rates decreased to %s \n" %(epoch, lr))
        thefile.write("Trained for %s epochs \n\n" %Parameters[0][-1])
        
        thefile.write("Train Statistics \n")
        thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
        thefile.write('MSE Average Loss: %s \n\n'%trainloss_to_file[-1])
        
        thefile.write("Test Statistics \n")
        thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
        thefile.write('MSE Average Loss: %s \n\n'%testloss_to_file[-1])

#        for i in range(len(trainloss_to_file)):
#            if trainloss_to_file[i]==trainloss_to_file[-1]:
#                break
#        if i+1==len(trainloss_to_file):
#            i=-1
#        thefile.write('Minimum loss accuracy in epoch %s (if 0  it means that the initial state was the best)\n\n'%str(i+1))
#        
        thefile.write('Total time elapsed %s\n\n' %Parameters[4])

        thefile.write(20*'-'+'\n\n')


        thefile.close() 
'''
########################################################################################################

def checkpoint_save(model,trainAcc_to_file,testAcc_to_file,trainloss_to_file,testloss_to_file,Parameters,Model_name,Train_mode,Dataset):
        
        path='CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset
        if not os.path.exists(path):
            os.makedirs(path)
            
        torch.save(model.state_dict(),path+'/CHECKPOINT.t7')
        if verborrea : print(path+'/CHECKPOINT.t7'+' saved')  

        thefile = open(path+'/Testacc_CHECKPOINT.csv', 'w')
        for item in testAcc_to_file:
            thefile.write("%s," % item)
        thefile.close()
    
        
        thefile = open(path+'/Testloss_CHECKPOINT.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Trainloss_CHECKPOINT.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
        thefile = open(path+'/Trainacc_CHECKPOINT.csv', 'w')
        for item in trainAcc_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Parameters_CHECKPOINT.txt', 'w')
            
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))       
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_grooups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
            
        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum learning rate decreased to %s \n" %(epoch, lr))
        if not(Parameters[0]==[]):
            thefile.write("Trained for %s epochs \n" %Parameters[0][-1])  
        thefile.write("\n\n" )    
        if not(trainAcc_to_file==[]):
            thefile.write("Train Statistics \n")
            thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
            thefile.write("Test Statistics \n")
            thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
            thefile.write(20*'-'+'\n\n')
        thefile.close() 


######################################################################################################
        
def checkpoint_save_stage1(model,trainloss_to_file,testloss_to_file,Parameters,Model_name,Train_mode,Dataset,fnum):
        
        path='CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+Dataset
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            model.module.features1
            parameters_str='model.module.features'
        except:
            parameters_str='model.features'
        feature_stateDict = parameters_str+str(fnum)+'.state_dict()'
        exec("torch.save(eval(feature_stateDict),path+'/CHECKPOINT_'+str(fnum)+'.t7')")
        if verborrea :print(path+'/CHECKPOINT_'+str(fnum)+'.t7')  

#        thefile = open('CHECKPOINT/Testacc_CHECKPOINT.csv', 'w')
#        for item in testAcc_to_file:
#            thefile.write("%s," % item)
#        thefile.close()
    
        
        thefile = open(path+'/Testloss_CHECKPOINT_feature'+str(fnum)+'.csv', 'w')
        for item in testloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        thefile = open(path+'/Trainloss_CHECKPOINT_feature'+str(fnum)+'.csv', 'w')
        for item in trainloss_to_file:
            thefile.write("%s," % item)
        thefile.close() 
        
        
#        thefile = open('CHECKPOINT/Trainacc_CHECKPOINT.csv', 'w')
#        for item in trainAcc_to_file:
#            thefile.write("%s," % item)
#        thefile.close() 
        
        thefile = open(path+'/Parameters_CHECKPOINT'+str(fnum)+'.txt', 'w')
        thefile.write("STAGE1 \n" )     
        thefile.write("Patience_scheduler=%s,  Weight_decay=%s  \n" %(Parameters[2],Parameters[3]))
        
        if not Parameters[1][0][1:] == Parameters[1][0][:-1]:
            for i in range(len(Parameters[1][0])):
                thefile.write("Initial learning rate for param_groups %s is %s epochs \n" %(str(i),Parameters[1][0][i]))
        else:
            thefile.write("Initial learning rate is %s epochs \n" %Parameters[1][0][0])
            
        for epoch,lr in zip(Parameters[0],Parameters[1][1:]):
            thefile.write("In epoch %s, maximum learning rate decreased to %s \n" %(epoch, lr))
        if not(Parameters[0]==[]):
            thefile.write("Trained for %s epochs \n" %Parameters[0][-1])  
        thefile.write("\n\n" )    
        if not(trainloss_to_file==[]):
#            thefile.write("Train Statistics \n")
#            thefile.write('Accuracy: %s \n' %trainAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%trainloss_to_file[-1])
        
            thefile.write("Test Statistics \n")
#            thefile.write('Accuracy: %s \n' %testAcc_to_file[-1])
            thefile.write('Average Loss: %s \n\n'%testloss_to_file[-1])
            thefile.write(20*'-'+'\n\n')
        thefile.close() 
