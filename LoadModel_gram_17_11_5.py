# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:20:08 2017

@author: user
"""

import torchvision
import torch.nn as nn
import Load_Save_Model as load
import sys
import torch
class LoadModel:
    
    def __init__(self,model_name,Dataset,train_mode='pyramid',bn=False,Teacher=False):
        self.model_name=model_name
        self.train_mode= train_mode
        self.preweight= False
        self.batch_norm = bn
        self.model=None
        self.Dataset=Dataset
        if 'pyramid' in train_mode.lower():
            self.preweight= True
            
        if model_name.upper()=='VGG19':
            self.model=VGG19(self.Dataset,self.train_mode)
            self.model=load.load_model(self.model,model_name.upper(),'SCRATCH',Dataset)
        elif model_name.upper()=='SVGG17':
            self.model=SVGG17(self.Dataset,self.train_mode,Teacher=Teacher)
        elif model_name.upper()=='SVGG14':
            self.model=SVGG14(self.Dataset,self.train_mode,self.preweight,Teacher=Teacher)
        elif model_name.upper()=='SVGG11':
            self.model=SVGG11(self.Dataset,self.train_mode,self.preweight,Teacher=Teacher)
        elif model_name.upper()=='SVGG8':
            self.model=SVGG8(self.Dataset,self.train_mode,self.preweight,Teacher=Teacher)
        elif model_name.upper()=='SVGG5':
            self.model=SVGG5(self.Dataset,self.train_mode,self.preweight,Teacher=Teacher)
        else:
            print('Load Model error!')
            sys.exit()
        if model_name!='VGG19':
            self.model=load.load_model(self.model,model_name.upper(),train_mode,Dataset)
    
            
#def VGG19(dataSet): 
#    vgg19 = torchvision.models.vgg19(pretrained=True)
#            
#    if(dataSet.upper()=='CIFAR100'):
#        vgg19.classifier._modules['0'] = nn.Linear(512, 512)
#        vgg19.classifier._modules['3'] = nn.Linear(512, 256)
#        vgg19.classifier._modules['6'] = nn.Linear(256, 100)
#        print('VGG19 (from torchvision.models) for scratch training loaded')
#    elif(dataSet.upper()=='CIFAR10'):
#        vgg19.classifier._modules['0'] = nn.Linear(512, 512)
#        vgg19.classifier._modules['3'] = nn.Linear(512, 256)
#        vgg19.classifier._modules['6'] = nn.Linear(256, 10)
#        print('VGG19 (from torchvision.models) for scratch training loaded')
#    elif(dataSet.upper()=='STL10'):
#        vgg19.classifier._modules['0'] = nn.Linear(4608, 2048)
#        vgg19.classifier._modules['3'] = nn.Linear(2048, 512)
#        vgg19.classifier._modules['6'] = nn.Linear(512, 10)
#        print('VGG19 (from torchvision.models) for scratch training loaded')
#    else:
#        print('Error in VGG19: Data set not found')
#        
#    return vgg19
            
def create_Gram_mat(F1,F2):
    F2=torch.nn.functional.upsample(F2,scale_factor=2,mode='nearest') #Added upsample code - Ashis
#    if F1.shape[0,2,3]!=F2.shape[0,2,3]:
#        F2=up(F2)
    if F1.size()[2]!=F2.size()[2] and F1.size()[3]!=F2.size()[3]:
        F2=torch.nn.functional.upsample(F2,scale_factor=2,mode='nearest')

    assert (F1.size()[0]==F2.size()[0] and F1.size()[2]==F2.size()[2] and F1.size()[3]==F2.size()[3])
    
    n=F1.size()[1]
    m=F2.size()[1]
    leng_batch=F1.size()[0]
    G=torch.zeros((leng_batch,n,m))
    if torch.cuda.is_available():
        G=torch.autograd.Variable(G.cuda())
    else:
        G=torch.autograd.Variable(G)
    h=len(F1[0,0,:,0])
    w=len(F1[0,0,0,:])
    
    '''
    for k in range(leng_batch):
        features1=F1[k,:,:,:].view(n, h * w)
        features2=F2[k,:,:,:].view(m, h * w)
        G[k,:,:]=torch.mm(features1,features2.t())
    G=G/(h*w)
    '''
    features1=F1.view(leng_batch,n, h * w)
    features2=F2.view(leng_batch,m, h * w)  
    features2=features2.permute(0,2,1)
    G=torch.bmm(features1,features2)/(h*w)
#    for k in range(leng_batch):
#        for i in range(n):
#            for j in range(m):
#                G[k,i,j]=torch.dot(F1[k,i,:,:].view(h*w),F2[k,j,:,:].view(h*w))
#        
#   G=G/(h*w)
    return G
        
class VGG19(nn.Module):

    def __init__(self,dataSet,Train_mode,Teacher=False,Stage=2):
        super(VGG19, self).__init__()
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=Train_mode
        original_model = torchvision.models.vgg19(pretrained=True)
        self.features=nn.Sequential(*list(original_model.features.children())[:37]) 
        self.classifier=nn.Sequential(*list(original_model.classifier.children())[:])
        if(dataSet.upper()=='CIFAR100'):
            self.classifier._modules['0'] = nn.Linear(512, 512)
            self.classifier._modules['3'] = nn.Linear(512, 256)
            self.classifier._modules['6'] = nn.Linear(256, 100)
            print('VGG19 (from torchvision.models) for scratch training loaded')
        elif(dataSet.upper()=='CIFAR10'):
            self.classifier._modules['0'] = nn.Linear(512, 512)
            self.classifier._modules['3'] = nn.Linear(512, 256)
            self.classifier._modules['6'] = nn.Linear(256, 10)
            print('VGG19 (from torchvision.models) for scratch training loaded')
        elif(dataSet.upper()=='STL10'):
            self.classifier._modules['0'] = nn.Linear(4608, 2048)
            self.classifier._modules['3'] = nn.Linear(2048, 512)
            self.classifier._modules['6'] = nn.Linear(512, 10)
            print('VGG19 (from torchvision.models) for scratch training loaded')
        elif(dataSet.upper()=='SVHN'):
            self.classifier._modules['0'] = nn.Linear(512, 512)
            self.classifier._modules['3'] = nn.Linear(512, 256)
            self.classifier._modules['6'] = nn.Linear(256, 10)
            print('VGG19 (from torchvision.models) for scratch training loaded')
        else:
            print('Error in VGG19: Data set not found')  
    
    def forward(self, x):
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            i=0
            for name,feature in self.features._modules.items():
                x=feature(x)
                if i==22:
                    break
                i=i+1
            return x
        
        
        x1=self.features(x)
        x2=x1.view(-1, self.num_flat_features(x1))
        x3=self.classifier(x2)
        
        return x3
            
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
            
            
class SVGG17(nn.Module):

    def __init__(self,dataSet,Train_mode,Teacher=False,Stage=2):  
        super(SVGG17, self).__init__()
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=Train_mode
        if 'gram' in self.train_mode.lower():
            original_model = SVGG17(dataSet,'pyramid')
            original_model=load.load_model(original_model,'SVGG17','pyramid',self.dataSet) 
            #removed check variable --Ashis
            self.compressed_features=nn.Sequential()
            self.frozen_features=nn.Sequential()
            self.features1=nn.Sequential(*list(original_model.features.children())[:2])
            self.features2=nn.Sequential(*list(original_model.features.children())[2:18])
            self.max_pool1=nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1))
            self.features3=nn.Sequential(*list(original_model.features.children())[19:36])
            self.max_pool2=nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1))
            self.classifier=nn.Sequential(*list(original_model.classifier.children())[:])
        else:
            original_model = VGG19(dataSet,Train_mode)
            original_model=load.load_model(original_model,'VGG19','scratch',self.dataSet) #removed check variable --Ashis
                #print('VGG19_CIFAR100_scratch.t7 file loaded')
#        if(check is None):
#                print('VGG19 couldnt be loaded. Exiting now...')
#                sys.exit()
                
            if 'fitnet' in self.train_mode.lower():
                print('No need of Regressor_SVGG17 for fitnet')
                
            self.compressed_features=nn.Sequential()
            self.frozen_features=nn.Sequential()
            self.features=nn.Sequential(*list(original_model.features.children())[:37])
            if dataSet == 'STL10': #Code added for STL10 databse --Ashis
                self.classifier=nn.Sequential(nn.Dropout(p=0.5),nn.Linear(4608,10))
            else:
                self.classifier=nn.Sequential(nn.Dropout(p=0.5),nn.Linear(512,100))
    def forward(self, x):
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            #x4=self.features[:21](x)
            i=0
            for name,feature in self.features._modules.items():
                x=feature(x)
                if i==20:
                    break
                i=i+1
            return x
        #Added code for gram matrix calculation
        if 'gram_pyramid' in self.train_mode.lower() and self.stage==1:
            f1 =self.features1(x)
            f2 =self.features2(f1)
            f2_m=self.max_pool1(f2)
            f3=self.features3(f2_m)
            gram = create_Gram_mat(f2_m,f3)
            return gram
        
        x4=self.features(x)
        
        if self.stage==1 and self.teacher:
            return x4
        x5 = x4.view(-1, self.num_flat_features(x4))
        x6= self.classifier(x5)
        if self.stage==1 and not self.teacher:
             return x6
       
        return [x4,x6,x6]

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class SVGG14(nn.Module):

    def __init__(self,dataSet,train_mode,pre_weight_load=True,Teacher=False,Stage=2):
        
        super(SVGG14, self).__init__()
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=train_mode
        original_model =SVGG17(dataSet,train_mode)
        if pre_weight_load:
            original_model=load.load_model(original_model,'SVGG17',train_mode,self.dataSet)
        else:
            print('Caution: SVGG17 NO WEIGHTS loaded. Using VGG19 Weights...')
            
        if 'fitnet' in self.train_mode.lower():
            self.regressor=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                         nn.Conv2d(256,512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)))
            print('Regressor_SVGG14 loaded for fitnet')

        self.features=nn.Sequential(*list(original_model.features.children())[:28])
        self.compressed_features=nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                           nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
        self.frozen_features=nn.Sequential()
        self.classifier=nn.Sequential(*list(original_model.classifier.children())[:])

    def forward(self, x):
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            #x4=self.features[:21](x)
            i=0
            for name,froz_feature in self.features._modules.items():
                x=froz_feature(x)
                if i==15:
                    break
                i=i+1
            xr=self.regressor(x)
            return xr

        
        x3=self.features(x)
        if self.stage==1 and self.teacher:
            return x3
        x4=self.compressed_features(x3)
        if self.stage==1 and not self.teacher:
            return x4
        x5 = x4.view(-1, self.num_flat_features(x4))
        x6=self.classifier(x5)
        return [x3,x4,x6]

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class SVGG11(nn.Module):

    def __init__(self,dataSet,train_mode,pre_weight_load=True,Teacher=False,Stage=2):
        
        super(SVGG11, self).__init__()
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=train_mode
        if 'gram' in self.train_mode.lower():
            original_model = SVGG17(dataSet,self.train_mode)
            original_model=load.load_model(original_model,'SVGG17',self.train_mode,self.dataSet) 
            #removed check variable --Ashis
            self.frozen_features=nn.Sequential()
            self.features1=nn.Sequential(*list(original_model.features1.children())[:2])
            self.features2=nn.Sequential(*list(original_model.features2.children())[0:16])
            self.max_pool1=nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1))

            self.compressed_features=nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                               nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                               nn.Conv2d(512, 512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                           nn.ReLU(inplace=True))
            
            self.max_pool2=nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1))
            self.classifier=nn.Sequential(*list(original_model.classifier.children()))
        else:
            original_model =SVGG14(dataSet,train_mode,pre_weight_load)
            if pre_weight_load:
                original_model=load.load_model(original_model,'SVGG14',train_mode,self.dataSet)
            else:
                print('Caution: SVGG14 NO WEIGHTS loaded. Using VGG19 Weights...')
                
            if 'fitnet' in train_mode.lower():
                self.regressor=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                             nn.Conv2d(256,512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)))
                print('Regressor_SVGG11 loaded for fitnet')
                
            self.features=nn.Sequential(*list(original_model.features.children())[:19])
            
            self.compressed_features=nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                               nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
            
            self.frozen_features=nn.Sequential(*list(original_model.compressed_features.children())[:])
            self.classifier=nn.Sequential(*list(original_model.classifier.children()))

    def forward(self, x):
        
        if 'gram_pyramid' in self.train_mode.lower():
            f1 =self.features1(x)
            f2 =self.features2(f1)
            if self.teacher and self.stage==1:
                gram = create_Gram_mat(f1,f2)
                return gram
            f2_m=self.max_pool1(f2)
            f3=self.compressed_features(f2_m)
            if self.stage==1:
                gram = create_Gram_mat(f2_m,f3)
                return gram
            if self.stage==2:
                f4=self.max_pool2(f3)
                f5 = f4.view(-1, self.num_flat_features(f4))
                f6=self.classifier(f5)
                return f6
            
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            #x4=self.features[:21](x)
            i=0
            for name,froz_feature in self.features._modules.items():
                x=froz_feature(x)
                if i==13:
                    break
                i=i+1
            xr=self.regressor(x)
            return xr
        
        x2=self.features(x)
        if self.stage==1 and self.teacher:
            return x2
        x3 =  self.compressed_features(x2)
        if self.stage==1 and not self.teacher:
            return x3
        x4 =  self.frozen_features(x3)
        x5 = x4.view(-1, self.num_flat_features(x4))
        x6= self.classifier(x5)
        return [x2,x3,x6]

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class SVGG8(nn.Module):

    def __init__(self,dataSet,train_mode,pre_weight_load=True,Teacher=False,Stage=2):
        super(SVGG8, self).__init__()
        original_model =SVGG11(dataSet,train_mode,pre_weight_load)
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=train_mode
        if pre_weight_load:
            original_model=load.load_model(original_model,'SVGG11',train_mode,self.dataSet)
        else:
            print('Caution: SVGG11 NO WEIGHTS loaded. Using VGG19 Weights...')
        
        if 'fitnet' in train_mode.lower():
            self.regressor=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                         nn.Conv2d(128,512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)))
            print('Regressor_SVGG8 loaded for fitnet')
            
            
        self.features=nn.Sequential(*list(original_model.features.children())[:10])
        self.compressed_features=nn.Sequential(nn.Conv2d(128,256, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                           nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
        self.frozen_features=nn.Sequential(*list(original_model.compressed_features.children())[:],*list(original_model.frozen_features.children())[:])
        self.classifier=nn.Sequential(*list(original_model.classifier.children()))

    def forward(self, x):
        
        x2=self.features(x)
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            xr=self.regressor(x2)
            return xr
        
        
        if self.stage==1 and self.teacher:
            return x2
        x3 =  self.compressed_features(x2)
        if self.stage==1 and not self.teacher:
            return x3
        x4 =  self.frozen_features(x3)
        x5 = x4.view(-1, self.num_flat_features(x4))
        x6= self.classifier(x5)
        return [x2,x3,x6]
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
 
class SVGG5(nn.Module):

    def __init__(self,dataSet,train_mode,pre_weight_load=True,Teacher=False,Stage=2):
        
        super(SVGG5, self).__init__()
        self.dataSet = dataSet
        self.stage=Stage
        self.teacher=Teacher
        self.train_mode=train_mode
        if 'gram' in self.train_mode.lower():
            original_model = SVGG11(dataSet,self.train_mode)
            original_model=load.load_model(original_model,'SVGG11',self.train_mode,self.dataSet) 
            #removed check variable --Ashis
            self.features1=nn.Sequential(*list(original_model.features1.children())[:2])
            self.max_pool1=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
            self.compressed_features=nn.Sequential(nn.Conv2d(64,128, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                               nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                               nn.Conv2d(128,256, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                           nn.ReLU(inplace=True))
            self.max_pool2=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
            self.frozen_features=nn.Sequential(*list(original_model.compressed_features.children()))
            
            self.max_pool3=nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1))
            self.classifier=nn.Sequential(*list(original_model.classifier.children()))
        else:
            original_model =SVGG8(dataSet,train_mode,pre_weight_load)
            if pre_weight_load:
                original_model=load.load_model(original_model,'SVGG8',train_mode,self.dataSet)
            else:
                print('Caution: SVGG8 NO WEIGHTS loaded. Using VGG19 Weights...')
            if 'fitnet' in train_mode.lower():
                self.regressor=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                                             nn.Conv2d(256,512, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)))
                print('Regressor_SVGG5 loaded for fitnet')
                
                                             
                                             
            self.features=nn.Sequential(*list(original_model.features.children())[:2])
            self.compressed_features=nn.Sequential(nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)),
                    nn.Conv2d(64,128, kernel_size=(3, 3),stride=(1, 1), padding=(1, 1)),
                                               nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2), dilation=(1, 1)))
            self.frozen_features=nn.Sequential(*list(original_model.compressed_features.children())[:],*list(original_model.frozen_features.children())[:])
            self.classifier=nn.Sequential(*list(original_model.classifier.children()))
        

    def forward(self, x,output_all=True):
        if 'gram_pyramid' in self.train_mode.lower():
            f1 =self.features1(x)
            f1_m=self.max_pool1(f1)
            f2 =self.compressed_features(f1_m)
            
            
            if self.stage==1:
                gram = create_Gram_mat(f1,f2)
                return gram
            if self.stage==2:
                f2_m=self.max_pool2(f2)
                f3=self.frozen_features(f2_m)
                f3_m=self.max_pool3(f3)
                f4 = f3_m.view(-1, self.num_flat_features(f3_m))
                f5=self.classifier(f4)
                
                return f5
        
        x2=self.features(x)
        
        
        if self.stage==1 and self.teacher and not 'fitnet' in self.train_mode.lower():
            return x2
        x3 =  self.compressed_features(x2)
        if self.stage==1 and not self.teacher and not 'fitnet' in self.train_mode.lower():
            return x3
        
        if 'fitnet' in self.train_mode.lower() and self.stage==1:
            #x4=self.features[:21](x)
            i=0
            for name,froz_feature in self.frozen_features._modules.items():
                x3=froz_feature(x3)
                if i==1:
                    break
                i=i+1
            xr=self.regressor(x3)
            return xr
        x4 =  self.frozen_features(x3)
        
        x5 = x4.view(-1, self.num_flat_features(x4))
        x6= self.classifier(x5)
        return [x2,x3,x6]

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features   

class VGGNetNthLayer(nn.Module):
            def __init__(self,dataset,n_layers,model_name):
                super(VGGNetNthLayer, self).__init__()
                if model_name[-3:]=='_bn':
                    batch_norm='_bn'
                else:
                    batch_norm=''
                self.original_model=VGG19(dataset,train_mode='pyramid')
                    
                self.original_model=LoadModel('VGG19'+batch_norm,dataset,'SCRATCH').model

                print(self.original_model)
                for param in self.original_model.parameters():
                    param.requires_grad = False
                
#models.vgg19(pretrained=True)#.cuda()
                #self.original_model = torch.nn.DataParallel(original_model, device_ids=range(torch.cuda.device_count()))
                #cudnn.benchmark = True
                self.features=[]

                self.features.append( nn.Sequential(*list(self.original_model.features.children())[:n_layers[0]]))

                for i in range(1,len(n_layers)):
                    self.features.append( nn.Sequential(*list(self.original_model.features.children())[n_layers[i-1]+1:n_layers[i]]))
                
                self.features.append(nn.Sequential(*list(self.original_model.classifier.children())))
            def forward(self, x):
                #print('VGG19_forward method')
                output=[]
                input_=self.features[0](x)
                output.append(input_)
                for feature in self.features[1:]:
                    input_=feature(input_)
                    output.append(input_)
                return output  
def Load_BigModel(model_name,train_mode,dataset):
    train_mode =train_mode.upper()
    if train_mode=='SCRATCH':
        big_model_name =''
        ensembleModel=None
    elif 'FITNET'==train_mode:
        big_model_name='VGG19'
        ensembleModel=LoadModel(big_model_name,dataset,'FITNET').model
    elif 'GRAM_PYRAMID'== train_mode:
        if model_name=='SVGG11':
            big_model_name='SVGG17'
            ensembleModel=LoadModel(big_model_name,dataset,train_mode).model
        elif model_name=='SVGG5':
            big_model_name='SVGG11'
            ensembleModel=LoadModel(big_model_name,dataset,train_mode,Teacher=True).model
    else:
        if 'PYRAMID' in train_mode:
            if model_name=='SVGG17':
                big_model_name='VGG19'
                ensembleModel=LoadModel(big_model_name,dataset,'SCRATCH').model
            else:
                if model_name=='SVGG14':
                    big_model_name='SVGG17'
                    ensembleModel=LoadModel(big_model_name,dataset,train_mode,Teacher=True).model
                else:
                    if model_name=='SVGG11':
                        big_model_name='SVGG14'
                        ensembleModel=LoadModel(big_model_name,dataset,train_mode,Teacher=True).model
                    else:
                        if model_name=='SVGG8':
                            big_model_name='SVGG11'
                            ensembleModel=LoadModel(big_model_name,dataset,train_mode,Teacher=True).model
                        else:
                            if model_name=='SVGG5':
                                big_model_name='SVGG8'
                                ensembleModel=LoadModel(big_model_name,dataset,train_mode,Teacher=True).model
    if ensembleModel!=None:
        for param in ensembleModel.parameters():
            param.requires_grad = False
    return big_model_name,ensembleModel
