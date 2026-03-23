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
import binaryconnect



# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Transforms and Data
# =========================
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

rootdir = '/opt/img/effdl-cifar10/'
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

trainloader = DataLoader(c10train, batch_size=64, shuffle=True)
testloader = DataLoader(c10test, batch_size=64)

# =========================
# WANDB INIT
# =========================
wandb.login()

run = wandb.init(
    
    entity="scherpereelant-imt-atlantique",
    project="efficient deep learning Resnet18",
    config={
        "learning_rate": 0.1,
        "architecture": "ResNet18 - BinaryConnect",
        "dataset": "CIFAR-10",
        "epochs": 55,
        "batch_size": 64,
    },
)

# =========================
# Utils
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100. * correct / total

def train_model(modelbc, epochs, run=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        modelbc.model.parameters(),
        lr=run.config["learning_rate"] if run else 0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    for epoch in range(epochs):
        modelbc.model.train()
        running_loss = 0.0
        for x, y in trainloader:
            
            modelbc.binarization() 
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = modelbc.forward(x)
            modelbc.restore()

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            modelbc.clip()
            running_loss += loss.item()
            
            
        avg_loss = running_loss / len(trainloader)
        modelbc.model.eval()
        modelbc.binarization() 
        acc_train = evaluate(modelbc.model, trainloader)
        acc_test = evaluate(modelbc.model, testloader)
        modelbc.restore() 

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc_train:.2f}% - Test Acc: {acc_test:.2f}%")
        if run is not None:
            run.log({"epoch": epoch+1, "loss": avg_loss, "train_acc": acc_train, "test_acc": acc_test, "lr": optimizer.param_groups[0]['lr']})
    return modelbc

# =========================
# Model
# =========================
model = ResNet18()
print(f"Number of parameters: {count_parameters(model)/1e6:.2f}M (ResNet18)")
modelbc = binaryconnect.BC(model)
modelbc.model = modelbc.model.to(device) # it has to be set for GPU training

# =========================
# Training
# =========================
modelbc = train_model(modelbc, epochs=run.config["epochs"], run=run)

# =========================
# Save model
# =========================
state = {
    'net': modelbc.model.state_dict(),
    'params': count_parameters(modelbc.model),
    'epochs': run.config["epochs"],
    'architecture': 'ResNet18 -binary',
    'dataset': 'CIFAR-10',
}
torch.save(state, 'checkpoints/ResNet18_binary.pth')
print("Model saved to checkpoints/ResNet18_binary.pth")

run.finish()
