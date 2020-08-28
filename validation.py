# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 09:40:46 2020

@author: Jianxiang Wang
"""
from sklearn import metrics
import torch
from models import *
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load

#define the net
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VGG('VGG16', 10)
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = LeNet()
# net = EfficientNetB0()
# net = LeNet2()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load('./checkpoint/ckpt.pth'))
net = net.module


#loading data
_, _, valloader, classes = load()


def validation():
    print(net.classifier)
    #print(net)
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
    # Confusion Matrix
    print("Confusion Matrix...")
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    Accuracy = 100.*correct/total

    predicted = predicted.cpu().numpy()
    targets = targets.data.cpu().numpy()
    cm = metrics.confusion_matrix(targets, predicted)
    print(cm)
    print('Accuracy=',Accuracy,"%")
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='Blues')

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

if __name__=='__main__':
    validation()


