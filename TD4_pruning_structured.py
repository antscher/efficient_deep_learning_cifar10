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
c10train = CIFAR10(rootdir, train=True,  download=True, transform=transform_train)
c10test  = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

trainloader = DataLoader(c10train, batch_size=64, shuffle=True)
testloader  = DataLoader(c10test,  batch_size=64)

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
            total   += y.size(0)
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

    # Evaluate before fine-tuning
    print("Evaluate before fine-tuning...")
    acc = evaluate(model, testloader)
    print(f"Accuracy before fine-tuning: {acc:.2f}%")
    if run is not None:
        run.log({"epoch": 0, "test_acc": acc})

    for epoch in range(1, epochs + 1):
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
        avg_loss  = running_loss / len(trainloader)
        acc_train = evaluate(model, trainloader)
        acc_test  = evaluate(model, testloader)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc_train:.2f}% - Test Acc: {acc_test:.2f}%")
        if run is not None:
            run.log({"epoch": epoch, "loss": avg_loss, "train_acc": acc_train,
                     "test_acc": acc_test, "lr": optimizer.param_groups[0]["lr"]})
    return model


# =========================
# Structured Pruning – Filter L1 norm (Li et al., ICLR 2017)
# arXiv:1608.08710
#
# Principle: rank filters by their L1 norm (sum of absolute weight values).
# Filters with the smallest L1 norm contribute the least to the output and
# are removed entirely (out_channels of conv_i, in_channels of conv_i+1, BN_i).
# This results in a physically smaller model with real FLOPs/param savings.
# =========================

def l1_norm_filter(conv: nn.Conv2d) -> torch.Tensor:
    """
    Compute the L1 norm of each filter in a Conv2d layer.
    Filter i has shape (C_in, kH, kW) → scalar norm.
    Returns a tensor of shape (C_out,).
    """
    # weight shape: (C_out, C_in, kH, kW)
    return conv.weight.data.abs().sum(dim=[1, 2, 3])


def prune_conv_out_channels(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new Conv2d keeping only the filters at keep_idx (output channels)."""
    new_conv = nn.Conv2d(
        conv.in_channels, len(keep_idx),
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=(conv.bias is not None)
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep_idx])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep_idx])
    return new_conv


def prune_conv_in_channels(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new Conv2d keeping only the input channels at keep_idx."""
    new_conv = nn.Conv2d(
        len(keep_idx), conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=(conv.bias is not None)
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[:, keep_idx])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def prune_batchnorm(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    """Create a new BatchNorm2d keeping only the channels at keep_idx."""
    new_bn = nn.BatchNorm2d(len(keep_idx), eps=bn.eps, momentum=bn.momentum,
                            affine=bn.affine, track_running_stats=bn.track_running_stats)
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight[keep_idx])
            new_bn.bias.copy_(bn.bias[keep_idx])
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean[keep_idx])
            new_bn.running_var.copy_(bn.running_var[keep_idx])
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def apply_structured_pruning(model, prune_ratio=0.5, verbose=True):
    """
    Structured filter pruning on ResNet18 following Li et al. (arXiv:1608.08710).

    For every conv1 inside each BasicBlock:
      1. Compute the L1 norm of each filter.
      2. Keep the top (1 - prune_ratio) filters (highest L1 norm = most important).
      3. Physically remove the pruned filters:
           - Replace conv1 with a smaller Conv2d (fewer output channels).
           - Replace bn1   with a smaller BatchNorm2d.
           - Replace conv2 with a smaller Conv2d (fewer input channels).

    We only prune conv1 (not conv2) to avoid mismatches with residual connections.

    Args:
        model      : ResNet18 instance (modified in-place).
        prune_ratio: Fraction of filters to remove per layer.
        verbose    : Print per-layer statistics.
    Returns:
        model: the pruned model.
    """
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            conv1 = block.conv1
            bn1   = block.bn1
            conv2 = block.conv2

            n_filters = conv1.out_channels
            n_keep    = max(1, int(round(n_filters * (1.0 - prune_ratio))))

            # --- Rank filters by L1 norm, keep the largest ---
            norms    = l1_norm_filter(conv1)                   # (C_out,)
            keep_idx = torch.argsort(norms, descending=True)[:n_keep]
            keep_idx, _ = keep_idx.sort()                      # keep original order

            if verbose:
                print(f"  Block conv1 ({n_filters} filters) → keep {n_keep} "
                      f"(prune {n_filters - n_keep}, {(n_filters-n_keep)/n_filters*100:.1f}%)")

            # --- Replace layers with smaller versions ---
            block.conv1 = prune_conv_out_channels(conv1, keep_idx)
            block.bn1   = prune_batchnorm(bn1, keep_idx)
            block.conv2 = prune_conv_in_channels(conv2, keep_idx)

    return model


# =========================
# Main pipeline
# =========================
if __name__ == "__main__":
    for PRUNE_RATIO in [ 0.6, 0.7, 0.8, 0.9]:
        print(f"\n=== Structured pruning with prune_ratio={PRUNE_RATIO} ===")
        FINETUNE_EPOCHS = 10
        CHECKPOINT_IN   = "checkpoints/ResNet18_mixup_cos.pth"
        CHECKPOINT_OUT  = f"checkpoints/ResNet18_structured_{int(PRUNE_RATIO*100)}.pth"

        # --- Load pre-trained model ---
        print("Loading pre-trained ResNet18...")
        loaded_cpt = torch.load(CHECKPOINT_IN, map_location=device)
        model = ResNet18()
        model.load_state_dict(loaded_cpt["net"])
        model = model.to(device)

        params_before = count_parameters(model)
        acc_before    = evaluate(model, testloader)
        print(f"Before pruning – params: {params_before:,}  acc: {acc_before:.2f}%\n")

        # --- Structured pruning (L1 filter norm) ---
        print(f"Applying structured L1-filter pruning (ratio={PRUNE_RATIO})...")
        model = apply_structured_pruning(model, prune_ratio=PRUNE_RATIO, verbose=True)
        model = model.to(device)

        params_after = count_parameters(model)
        acc_after    = evaluate(model, testloader)
        print(f"\nAfter pruning  – params: {params_after:,}  acc: {acc_after:.2f}%")
        print(f"Compression: {params_before / params_after:.2f}x\n")

        # --- Fine-tune with WandB ---
        wandb.login()
        run = wandb.init(
            entity="scherpereelant-imt-atlantique",
            project="efficient deep learning Resnet18",
            config={
                "learning_rate":    0.01,
                "architecture":     f"ResNet18 structured pr={PRUNE_RATIO}",
                "dataset":          "CIFAR-10",
                "epochs":           FINETUNE_EPOCHS,
                "batch_size":       64,
                "prune_ratio":      PRUNE_RATIO,
                "params_before":    params_before,
                "params_after":     params_after,
                "acc_before_prune": acc_before,
                "acc_after_prune":  acc_after,
                "method":           "L1 filter norm (Li et al. 2017)",
            },
        )

        model = train_model(model, epochs=FINETUNE_EPOCHS, run=run)

        acc_final = evaluate(model, testloader)
        print(f"\nFinal accuracy after fine-tuning: {acc_final:.2f}%")
        
        
        run.log({"final_test_acc": acc_final})

        # --- Save ---
        torch.save({
            "net":              model.state_dict(),
            "params":           count_parameters(model),
            "prune_ratio":      PRUNE_RATIO,
            "architecture":     f"ResNet18 structured {int(PRUNE_RATIO*100)}%",
            "dataset":          "CIFAR-10",
            "acc_before_prune": acc_before,
            "acc_after_prune":  acc_after,
            "acc_final":        acc_final,
        }, CHECKPOINT_OUT)
        print(f"Saved to {CHECKPOINT_OUT}")

        wandb.finish()
