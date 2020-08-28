# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 09:40:46 2020

@author: Jianxiang Wang
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#using own dataset or not
own_dataset = False
train_data_folder = './Dataset/train/'
test_data_folder = './Dataset/test/'
val_data_folder = './Dataset/val/'

# Data_transform
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.Resize(32,32),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(1),      #for grayscale
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914,), (0.2023,))
    ])

transform_test = transforms.Compose([
    #transforms.Resize(32,32),
    transforms.Grayscale(1),      #for grayscale
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, ), (0.2023,))
    ])


transform_val = transforms.Compose([
    #transforms.Resize(32,32),
    transforms.Grayscale(1),      #for grayscale
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, ), (0.2023,))
    ])


def load():
    if (own_dataset):
        print('using custom dataset')
        trainset = ImageFolder(train_data_folder, transform_train)
        testset = ImageFolder(test_data_folder, transform_test)
        valset = ImageFolder(val_data_folder, transform_val)

        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    else:
        print('using public dataset')
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)  

        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, valloader, classes
        






