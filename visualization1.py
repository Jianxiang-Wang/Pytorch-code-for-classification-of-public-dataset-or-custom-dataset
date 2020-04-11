import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from models import *
import torch.backends.cudnn as cudnn
#from torchvision.datasets import ImageFolder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VGG('VGG19')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
net.load_state_dict(torch.load('./checkpoint/ckpt.pth'))
net = net.module
print(net.features)



transform_train = transforms.Compose([
    #transforms.Resize(224,224),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    #transforms.Resize(224,224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

 
# mean  先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
# 使用 ImageFolder 必须有对应的目录结构
"""
train = ImageFolder("./datas/dogs-vs-cats/train", simple_transform)
valid = ImageFolder("./datas/dogs-vs-cats/valid", simple_transform)
"""

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


 
# 提取不同层输出的 主要代码
class LayerActivations:
    features = None
 
    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
 
    def hook_fn(self, module, input, output):
        self.features = output.cpu()
 
    def remove(self):
        self.hook.remove()
 

 
conv_out = LayerActivations(net.features, 0)  # 提出第 一个卷积层的输出 
img = next(iter(trainloader))[0]
 

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
# imshow(img)
conv_out.remove()  #
act = conv_out  # act 即 第0层输出的特征
 
# 可视化 输出
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
#for i in range(30):
#    ax = fig.add_subplot(12, 5, i+1, xticks=[], yticks=[])
#    ax.imshow(act[0][i].detach().numpy(), cmap="gray")
 
plt.show()