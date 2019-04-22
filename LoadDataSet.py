# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:06:43 2017

@author: user
"""
import torch
import torchvision
import torchvision.transforms as transforms
import lsun_testload

class LoadDataSet:
    def __init__(self,dataset):
        
        if(dataset.upper()=='CIFAR10'):
            self.data= CIFAR10()
        elif(dataset.upper()=='CIFAR100'):
            self.data= CIFAR100()
        elif(dataset.upper()=='STL10'):
            self.data= STL10()
        elif(dataset.upper()=='LSUN'):
            self.data= LSUN_Dataset()
        elif(dataset.upper()=='SVHN'):
            self.data= SVHN()
        else:
            print('SORRY ! Requested dataset not found')
            


def LSUN_Dataset():
    print('==> Preparing LSUN Small Traindata..')
    transform_train = transforms.Compose([
    transforms.Scale((96,96)),
    #transforms.RandomCrop((256,256), padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5102, 0.4743, 0.4389), (0.2739, 0.2745, 0.2926))] #Mean and standard deviation is for LSUN SMALL
    )
    
    
 
    #   trainset = torchvision.datasets.ImageFolder('./data/LSUN/train_big/',transform=transform_train)
    '''
    totalsum1=0
    totalsum2=0
    totalsum3=0
    np1=0
    for  img,label in trainloader.dataset:
        n1 = torch.sum((img[0,:,:]-mean1)**2)
        n2 = torch.sum((img[1,:,:]-mean2)**2)
        n3 = torch.sum((img[2,:,:]-mean3)**2)
        
        np1 = np1+img.size()[1]*img.size()[2]
        totalsum1 = totalsum1+ n1
        totalsum2 = totalsum2+ n2
        totalsum3 = totalsum3+ n3
    import math
    std1 = math.sqrt(totalsum1 /(np1-1))
    print(std1)
    std2 = math.sqrt(totalsum2 /(np1-1))
    print(std2)
    std3 = math.sqrt(totalsum3 /(np1-1))
    print(std3)
    '''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
    print('==> Preparing Testdata..')
    transform_test = transforms.Compose([
    transforms.Scale((96,96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5102, 0.4743, 0.4389), (0.2739, 0.2745, 0.2926))])
        
    testset= lsun_testload.LSUN('./data/LSUN/','test',transform= transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
    
    return (trainloader, testloader)

def SVHN():
   
    print('==> Preparing SVHN Traindata..')
    transform_train = transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.6564, 0.6559, 0.6681), (0.2991, 0.3004, 0.2982))
    ])
    
    def target_transform(target):
        return int(target[0]) - 1
        
    #extra = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform_train,target_transform=target_transform)
    #train = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train,target_transform=target_transform)
    #trainset = torch.utils.data.ConcatDataset([extra,train])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train,target_transform=target_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    '''
    totalsum_mean1=0
    totalsum_mean2=0
    totalsum_mean3=0
    np1=0
    for  img,label in trainloader.dataset:
        n1 = torch.sum(img[0,:,:])
        n2 = torch.sum(img[1,:,:])
        n3 = torch.sum(img[2,:,:])
        np1 = np1+img.size()[1]*img.size()[2]
        totalsum_mean1 = totalsum_mean1+ n1
        totalsum_mean2 = totalsum_mean2+ n2
        totalsum_mean3 = totalsum_mean3+ n3
    import math
    mean1 = math.sqrt(totalsum_mean1 /(np1))
    print(mean1)
    mean2 = math.sqrt(totalsum_mean2 /(np1))
    print(mean2)
    mean3 = math.sqrt(totalsum_mean3 /(np1))
    print(mean3) 
    np1=0
    totalsum_std1=0
    totalsum_std2=0
    totalsum_std3=0
    for  img,label in trainloader.dataset:
        n1 = torch.sum((img[0,:,:]-mean1)**2)
        n2 = torch.sum((img[1,:,:]-mean2)**2)
        n3 = torch.sum((img[2,:,:]-mean3)**2)
        np1 = np1+img.size()[1]*img.size()[2]
        
        totalsum_std1 = totalsum_std1+ n1
        totalsum_std2 = totalsum_std2+ n2
        totalsum_std3 = totalsum_std3+ n3
    std1 = math.sqrt(totalsum_std1 /(np1-1))
    print(std1)
    std2 = math.sqrt(totalsum_std2 /(np1-1))
    print(std2)
    std3 = math.sqrt(totalsum_std3 /(np1-1))
    print(std3)
    '''
        
    print('==> Preparing Testdata..')
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.6564, 0.6559, 0.6681), (0.2991, 0.3004, 0.2982))])
        
    testset = torchvision.datasets.SVHN(root='./data',split='test', download=True, transform=transform_test,target_transform=target_transform)
    #extra = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform_test,target_transform=target_transform)
    #test = torchvision.datasets.SVHN(root='./data',split='test', download=True, transform=transform_test,target_transform=target_transform)
    #testset = torch.utils.data.ConcatDataset([extra,test])

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
    
    return (trainloader, testloader)

def CIFAR10():
   
    print('==> Preparing CIFAR10 Traindata..')
    transform_train = transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        
    print('==> Preparing Testdata..')
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
    
    return (trainloader, testloader)
   
def CIFAR100():
    print('==> Preparing Traindata..')
    transform_train = transform_train = transforms.Compose([
    transforms.Scale((96,96)),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    print('==> Preparing Testdata..')
    transform_test = transforms.Compose([
    transforms.Scale((96,96)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    return (trainloader, testloader)

def STL10():
    print('==> Preparing Traindata..')
    transform_train = transform_train = transforms.Compose([
    transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.STL10(root='./data',split='train',  download=True, transform=transform_train)    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    print('==> Preparing Testdata..')
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.STL10(root='./data',split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    return (trainloader, testloader)
