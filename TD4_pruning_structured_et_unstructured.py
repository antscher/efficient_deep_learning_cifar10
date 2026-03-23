# =========================
# Imports
# =========================
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb
from pytorch_cifar.models.resnet import ResNet10, ResNet12, ResNet14, ResNet16, ResNet18

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Transforms and Data
# =========================
normalize_scratch = transforms.Normalize(
    (0.49142, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
)
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize_scratch,
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize_scratch,
    ]
)

rootdir = "/opt/img/effdl-cifar10/"
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

trainloader = DataLoader(c10train, batch_size=64, shuffle=True)
testloader = DataLoader(c10test, batch_size=64)


# =========================
# Utils
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, half=False):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if half:
                x = x.half()  # Convert input to half precision
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def train_model(model, epochs, run=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=run.config["learning_rate"] if run else 0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    acc0 = evaluate(model, testloader)
    print(f"  [epoch 0 – before fine-tune] test acc: {acc0:.2f}%")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        acc_train = evaluate(model, trainloader)
        acc_test = evaluate(model, testloader)
        scheduler.step()
        print(
            f"  Epoch {epoch}/{epochs} – loss: {avg_loss:.4f} "
            f"– train: {acc_train:.2f}% – test: {acc_test:.2f}%"
        )
        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "train_acc": acc_train,
                    "test_acc": acc_test,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
    return model


# ============================================================
# STEP 1 – Structured pruning (filter L1 norm, Li et al. 2017)
# arXiv:1608.08710
#
# Removes entire filters (out-channels of conv1, corresponding
# channels in bn1 and in-channels of conv2) based on their L1
# norm.  Physically shrinks the weight tensors → real FLOPs/
# param reduction even without sparse libraries.
# ============================================================


def l1_norm_filter(conv: nn.Conv2d) -> torch.Tensor:
    # weight: (C_out, C_in, kH, kW) → one scalar per filter
    return conv.weight.data.abs().sum(dim=[1, 2, 3])


def _new_conv_out(conv, keep_idx):
    new = nn.Conv2d(
        conv.in_channels,
        len(keep_idx),
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[keep_idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[keep_idx])
    return new


def _new_conv_in(conv, keep_idx):
    new = nn.Conv2d(
        len(keep_idx),
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[:, keep_idx])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def _new_bn(bn, keep_idx):
    new = nn.BatchNorm2d(
        len(keep_idx),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    with torch.no_grad():
        if bn.affine:
            new.weight.copy_(bn.weight[keep_idx])
            new.bias.copy_(bn.bias[keep_idx])
        if bn.track_running_stats:
            new.running_mean.copy_(bn.running_mean[keep_idx])
            new.running_var.copy_(bn.running_var[keep_idx])
            new.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new


def apply_structured_pruning(model, prune_ratio=0.3, verbose=True):
    """
    Structured filter pruning on every conv1 inside ResNet18 BasicBlocks.
    Filters with the lowest L1 norm are physically removed.
    conv2 is left untouched (dimension must match the residual connection).
    """
    if verbose:
        print(f"\n── Structured pruning (ratio={prune_ratio}) ──")
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
            n_total = conv1.out_channels
            n_keep = max(1, int(round(n_total * (1.0 - prune_ratio))))

            norms = l1_norm_filter(conv1)
            keep_idx = torch.argsort(norms, descending=True)[:n_keep]
            keep_idx, _ = keep_idx.sort()

            if verbose:
                print(
                    f"  conv1 {n_total}→{n_keep} filters "
                    f"(removed {n_total - n_keep}, {(n_total - n_keep) / n_total * 100:.1f}%)"
                )

            block.conv1 = _new_conv_out(conv1, keep_idx)
            block.bn1 = _new_bn(bn1, keep_idx)
            block.conv2 = _new_conv_in(conv2, keep_idx)
    return model


# ============================================================
# STEP 2 – Unstructured pruning (global L1, standard PyTorch)
#
# After structured pruning the model is already smaller.
# We then apply weight-level L1 pruning across all Conv2d and
# Linear layers to further increase sparsity.  This uses
# PyTorch's prune API (mask-based); we make it permanent at
# the end so the state_dict is clean.
# ============================================================


def apply_unstructured_pruning(model, prune_ratio=0.7, verbose=True):
    """
    Global unstructured L1 pruning on all Conv2d and Linear layers.
    Returns the list of (module, 'weight') pairs so masks can be
    removed later.
    """
    if verbose:
        print(f"\n── Unstructured pruning (ratio={prune_ratio}) ──")
    modules_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    for module, param in modules_to_prune:
        prune.l1_unstructured(module, name=param, amount=prune_ratio)
    if verbose:
        total = sum(m.weight_mask.numel() for m, _ in modules_to_prune)
        nonzero = sum(m.weight_mask.sum().item() for m, _ in modules_to_prune)
        print(
            f"  Sparsity: {100 * (1 - nonzero / total):.1f}%  "
            f"({int(total - nonzero):,} / {total:,} weights zeroed)"
        )
    return modules_to_prune


def remove_pruning_masks(modules_to_prune):
    """Make unstructured pruning permanent (remove weight_orig / mask)."""
    for module, param in modules_to_prune:
        prune.remove(module, param)


# ============================================================
# Main pipeline
# ============================================================
if __name__ == "__main__":
    for model_str in ["ResNet10", "ResNet12", "ResNet14", "ResNet16"]:
        match model_str:
            case "ResNet10":
                model = ResNet10()
            case "ResNet12":
                model = ResNet12()
            case "ResNet14":
                model = ResNet14()
            case "ResNet16":
                model = ResNet16()
            case "ResNet18":
                model = ResNet18()
        UNSTRUCTURED_RATIO = 0.5
        STRUCTURED_RATIO = 0.8
        CHECKPOINT_OUT = f"checkpoints/{model_str}_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_{model_str}.pth"
        FINETUNE_EPOCHS = 25
        CHECKPOINT_IN = f"checkpoints/{model_str}_mixup_cos_SGD.pth"

        # ── Load ──────────────────────────────────────────────────────────────
        print(f"Loading pre-trained {model_str}...")
        ckpt = torch.load(CHECKPOINT_IN, map_location=device)
        model.load_state_dict(ckpt["net"])
        model = model.to(device)

        params_orig = count_parameters(model)
        acc_orig = evaluate(model, testloader)
        print(f"Original  – params: {params_orig:,}  acc: {acc_orig:.2f}%")

        # ── Step 1 : Structured pruning (filter L1 norm) ───────────────────
        model = apply_structured_pruning(model, prune_ratio=STRUCTURED_RATIO)
        model = model.to(device)

        params_after_struct = count_parameters(model)
        acc_after_struct = evaluate(model, testloader)
        print(
            f"\nAfter structured  – params: {params_after_struct:,}  "
            f"acc: {acc_after_struct:.2f}%  "
            f"({params_orig / params_after_struct:.2f}x compression)"
        )

        # ── Step 2 : Unstructured pruning (weight L1) ──────────────────────
        pruned_modules = apply_unstructured_pruning(
            model, prune_ratio=UNSTRUCTURED_RATIO
        )

        acc_after_unstruct = evaluate(model, testloader)
        print(f"\nAfter unstructured – acc: {acc_after_unstruct:.2f}%")

        # ── Step 3 : Fine-tune to recover accuracy ─────────────────────────
        wandb.login()
        run = wandb.init(
            entity="scherpereelant-imt-atlantique",
            project="efficient deep learning Resnet depth comparison",
            config={
                "learning_rate": 0.001,
                "architecture": f"{model_str}_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_{model_str}",
                "dataset": "CIFAR-10",
                "epochs": FINETUNE_EPOCHS,
                "batch_size": 64,
                "structured_ratio": STRUCTURED_RATIO,
                "unstructured_ratio": UNSTRUCTURED_RATIO,
                "params_orig": params_orig,
                "params_after_struct": params_after_struct,
                "acc_orig": acc_orig,
                "acc_after_struct": acc_after_struct,
                "acc_after_unstruct": acc_after_unstruct,
            },
        )

        print(f"\nFine-tuning for {FINETUNE_EPOCHS} epochs...")
        model = train_model(model, epochs=FINETUNE_EPOCHS, run=run)

        # Make unstructured pruning permanent before saving
        remove_pruning_masks(pruned_modules)

        acc_final = evaluate(model, testloader)
        print(f"\nFinal accuracy: {acc_final:.2f}%")
        run.log({"final_test_acc": acc_final})

        print("Quantatization in 16 bits...")
        model_half = model.half()
        acc_final = evaluate(model_half, testloader, half=True)
        print(f"Final accuracy after quantization: {acc_final:.2f}%")

        # ── Save ───────────────────────────────────────────────────────────
        # Full model object (recommended – reloadable without knowing the architecture)
        torch.save(model, CHECKPOINT_OUT.replace(".pth", "_full_model.pth"))

        # State-dict checkpoint
        torch.save(
            {
                "net": model.state_dict(),
                "params": count_parameters(model),
                "structured_ratio": STRUCTURED_RATIO,
                "unstructured_ratio": UNSTRUCTURED_RATIO,
                "architecture": f"{model_str}_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_{model_str}",
                "dataset": "CIFAR-10",
                "acc_orig": acc_orig,
                "acc_after_struct": acc_after_struct,
                "acc_after_unstruct": acc_after_unstruct,
                "acc_final": acc_final,
            },
            CHECKPOINT_OUT,
        )

        print(f"Saved to {CHECKPOINT_OUT}")
        wandb.finish()
