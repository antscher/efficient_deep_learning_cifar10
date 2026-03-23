# =========================
# Imports
# =========================
from asyncio import run
from math import floor
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
import torch.nn.utils.prune as prune


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Transforms and Data
# =========================
normalize_scratch = transforms.Normalize((0.49142, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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

def train_model(model, epochs, run=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=run.config["learning_rate"] if run else 0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)    
    for epoch in range(1, epochs+1):

        model.train()
        running_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        acc_train = evaluate(model, trainloader)
        acc_test = evaluate(model, testloader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc_train:.2f}% - Test Acc: {acc_test:.2f}%")
        if run is not None:
            run.log({"epoch": epoch+1, "loss": avg_loss, "train_acc": acc_train, "test_acc": acc_test, "lr": optimizer.param_groups[0]['lr']})
    return model

# We load the dictionary
# loaded_cpt = torch.load('checkpoints/ResNet18_mixup_cos.pth',map_location=device)


# Define the model
# model = ResNet18()

# Finally we can load the state_dict in order to load the trained parameters
# model.load_state_dict(loaded_cpt['net'])
# model = model.to(device)


# model.eval()

# =========================
# Unstructured Pruning on Every Layer
# =========================

def apply_global_unstructured_pruning(model, prune_ratio=0.5, verbose=True):
    """
    Apply unstructured L1 pruning to all Conv2d and Linear layers of a model.
    Args:
        model (nn.Module): The model to prune (in-place).
        prune_ratio (float): Fraction of weights to prune in each layer.
        verbose (bool): If True, prints pruning info.
    Returns:
        model (nn.Module): The pruned model (same instance, pruned in-place).
    """
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))
    for module, param in modules_to_prune:
        prune.l1_unstructured(module, name=param, amount=prune_ratio)
        if verbose:
            print(f"Applied unstructured L1 pruning with ratio {prune_ratio} to all Conv2d and Linear layers.")
     

        
    model = train_model(model, 2, run=run)
        
        
    # Remove pruning re-parametrization (make pruning permanent, remove mask/weight_orig)
    for module, param in modules_to_prune:
        prune.remove(module, param)
    if verbose:
        print("Removed pruning re-parametrization (mask/weight_orig) from all pruned layers.")
    # Save pruned model as a normal one, with prune ratio in filename
    state = {
        'net': model.state_dict(),
        'params': count_parameters(model),
        'epochs': run.config["epochs"],
        'architecture': f'ResNet18 - pruned {int(prune_ratio*100)}%',
        'dataset': 'CIFAR-10',
    }
    if verbose:
        print(f"Saved pruned model to checkpoints/ResNet18_pruned_{int(prune_ratio*100)}_pipeline.pth")
    torch.save(state, f'checkpoints/ResNet18_pruned_{int(prune_ratio*100)}_pipeline.pth')
    


    
            
        

# Example usage (remove or comment out if importing this file as a module):
if __name__ == "__main__":
    for prune_ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_cpt = torch.load('checkpoints/ResNet18_mixup_cos.pth',map_location=device)
        # Define the model
        model = ResNet18()
        #    Finally we can load the state_dict in order to load the trained parameters
        model.load_state_dict(loaded_cpt['net'])
        model = model.to(device)
        prune_intermediate_ratio = 0.1
        ratio = 0.0
        wandb.login()

        run = wandb.init(
                    
            entity="scherpereelant-imt-atlantique",
            project="efficient deep learning Resnet18",
            config={
                "learning_rate": 0.001,
                "architecture": f"ResNet18 pr : {prune_ratio} - fine _ pipeline",
                "dataset": "CIFAR-10",
                "epochs": 2,
                "batch_size": 64,
            },
        )

        while prune_ratio > ratio:
            #fine tune the pruned model for a few epochs to recover some accuracy
            
            ratio += min(prune_intermediate_ratio, prune_ratio - ratio)
            print(f"Applying pruning with intermediate ratio {ratio:.2f}...")
            apply_global_unstructured_pruning(model, ratio, verbose=False)

            acc = evaluate(model, testloader)
            print(f"Accuracy after pruning with ratio {ratio}: {acc:.2f}%")

        wandb.finish()