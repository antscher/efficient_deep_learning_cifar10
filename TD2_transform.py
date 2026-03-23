# =========================
# Imports
# =========================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import wandb
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pytorch_cifar.models.resnet import ResNet18

normalize_scratch = transforms.Normalize((0.49142, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2),  # moins de padding
    transforms.RandomHorizontalFlip(p=0.3),  # moins de flips
    transforms.RandomRotation(2),  # rotation très faible
    transforms.RandomGrayscale(p=0.03),  # très peu de grayscale
    transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01),  # effet très léger
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),  # translation/scale très faible
    transforms.ToTensor(),
    normalize_scratch,
])

rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)

trainloader = DataLoader(c10train,batch_size=4,shuffle=False) ### Shuffle to False so that we always see the same images

from matplotlib import pyplot as plt 

# Visualisation batch avec DA
f = plt.figure(figsize=(10,10))
for i,(data,target) in enumerate(trainloader):
    data = (data.numpy())
    print(data.shape)
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    break
f.savefig('train_DA.png')

# Visualisation batch sans DA
transform_noDA = transforms.Compose([
    transforms.ToTensor(),
])
c10train_noDA = CIFAR10(rootdir, train=True, download=False, transform=transform_noDA)
trainloader_noDA = DataLoader(c10train_noDA, batch_size=4, shuffle=False)
f2 = plt.figure(figsize=(10,10))
for i, (data, target) in enumerate(trainloader_noDA):
    data = data.numpy()
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))
    break
f2.savefig('train_noDA.png')


