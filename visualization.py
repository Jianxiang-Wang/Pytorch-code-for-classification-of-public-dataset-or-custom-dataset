import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models import *
import os
import numpy as np
import dataset
from dataset import load
from PIL import Image
import imageio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VGG('VGG16', 10)
net = net.to(device)

resize_weight = 32
resize_height = 32

image_dir = '/home/zju/文档/CSV2IMG/datasets/train/5/48.jpg'
feature_dir = './Feautures/'


# Data_transform
print('==> Preparing data..')
transform_visualzation = transforms.Compose([
    transforms.Resize((resize_weight,resize_height)),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914,), (0.2023,))
    ])

def get_image(image_dir):

    image = Image.open(image_dir).convert('RGB')
    image = transform_visualzation(image)
    image = image.unsqueeze(0)
    return image



def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index,layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x


def get_layer_num_feature_map(feature_extractor, x):
    with torch.no_grad():
        for index,layer in enumerate(feature_extractor):
            x = layer(x)
        return index
            

 
def show_feature_map(feature_map, save_dir):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index-1], cmap=plt.cm.gray_r)
        plt.axis('off')
        store_path = os.path.join(save_dir, str(index)+".png")
        imageio.imwrite(store_path, feature_map[index-1])
    #plt.show()



def show_all_feature_map(feature_extractor, x, save_feature_dir):
    layer_num = get_layer_num_feature_map(feature_extractor,x)
    for i in range(layer_num):
        file_name = save_feature_dir + str(i)
        os.mkdir(file_name)
        feature_map = get_k_layer_feature_map(feature_extractor, i, x)
        show_feature_map(feature_map, file_name)

image = get_image(image_dir)
image = image.to(device)
k = 0
feature_extractor = net.features
#feature_map = get_k_layer_feature_map(feature_extractor, k, image)
#show_feature_map(feature_map, feature_dir)
show_all_feature_map(feature_extractor, image, feature_dir)
