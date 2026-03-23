# =========================
# Imports
# =========================
import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision.transforms as transforms
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
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
os.makedirs("checkpoints", exist_ok=True)
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
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if half:
                x = x.half()
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def train_model(model, epochs, run=None, stage_name="train"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = run.config["learning_rate"] if run else 0.01
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    acc0 = evaluate(model, testloader)
    print(f"  [{stage_name} epoch 0 – before fine-tune] test acc: {acc0:.2f}%")

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
            f"  [{stage_name}] Epoch {epoch}/{epochs} – loss: {avg_loss:.4f} "
            f"– train: {acc_train:.2f}% – test: {acc_test:.2f}%"
        )
        if run is not None:
            run.log(
                {
                    f"{stage_name}/epoch": epoch,
                    f"{stage_name}/loss": avg_loss,
                    f"{stage_name}/train_acc": acc_train,
                    f"{stage_name}/test_acc": acc_test,
                    f"{stage_name}/lr": optimizer.param_groups[0]["lr"],
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
    Structured filter pruning on every conv1 inside ResNet BasicBlocks.
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
# STEP 3 – Fake Quantization (configurable k-bits)
#
# Apply simulated quantization to weights and biases without
# actually converting to lower precision. This allows fine-tuning
# with gradient flow while simulating quantization effects.
# ============================================================


def apply_fake_quantization(model, k_bits=8, verbose=True):
    """
    Apply fake quantization with k-bits precision.

    Args:
        model: PyTorch model to quantize
        k_bits: Number of bits for quantization (default: 8)
        verbose: Print quantization info

    Returns:
        model_quantized: Model with fake quantization applied
    """
    if verbose:
        print(f"\n── Fake quantization ({k_bits}-bit) ──")

    model_quantized = copy.deepcopy(model).to(device)
    model_quantized.eval()

    quant_min = 0
    quant_max = (2**k_bits) - 1

    fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=quant_min,
        quant_max=quant_max,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )()
    fake_quant = fake_quant.to(device)

    with torch.no_grad():
        for module in model_quantized.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data.copy_(fake_quant(module.weight.data))
                if module.bias is not None:
                    module.bias.data.copy_(fake_quant(module.bias.data))

    if verbose:
        print(f"  Quantization range: [{quant_min}, {quant_max}]")

    return model_quantized


# ============================================================
# Main pipeline: Pruning + Quantization
# ============================================================
if __name__ == "__main__":
    # ==================== Configuration ====================
    MODELS = ["ResNet10", "ResNet12", "ResNet14", "ResNet16"]

    # Pruning parameters
    STRUCTURED_RATIO = 0.3  # Fraction of filters to prune
    UNSTRUCTURED_RATIO = 0.5  # Fraction of weights to prune

    # Quantization parameters
    K_BITS = 8  # Number of bits for quantization (4, 8, 16)

    # Training parameters
    FINETUNE_EPOCHS_PRUNING = 25
    FINETUNE_EPOCHS_QUANT = 20

    # ========================================================

    wandb.login()

    for model_str in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Processing {model_str}")
        print(f"{'=' * 60}")

        # ── Load model ──────────────────────────────────────
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

        CHECKPOINT_IN = f"checkpoints/{model_str}_mixup_cos_SGD.pth"
        CHECKPOINT_OUT = f"checkpoints/{model_str}_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_quant{K_BITS}b.pth"

        print(f"Loading pre-trained {model_str}...")
        ckpt = torch.load(CHECKPOINT_IN, map_location=device)
        model.load_state_dict(ckpt["net"])
        model = model.to(device)

        params_orig = count_parameters(model)
        acc_orig = evaluate(model, testloader)
        print(f"Original  – params: {params_orig:,}  acc: {acc_orig:.2f}%")

        # ── Step 1: Structured pruning ──────────────────────
        print(f"\n[STEP 1] Applying structured pruning...")
        model = apply_structured_pruning(
            model, prune_ratio=STRUCTURED_RATIO, verbose=True
        )
        model = model.to(device)

        params_after_struct = count_parameters(model)
        acc_after_struct = evaluate(model, testloader)
        print(
            f"\nAfter structured  – params: {params_after_struct:,}  "
            f"acc: {acc_after_struct:.2f}%  "
            f"({params_orig / params_after_struct:.2f}x compression)"
        )

        # ── Step 2: Unstructured pruning ────────────────────
        print(f"\n[STEP 2] Applying unstructured pruning...")
        pruned_modules = apply_unstructured_pruning(
            model, prune_ratio=UNSTRUCTURED_RATIO, verbose=True
        )

        acc_after_unstruct = evaluate(model, testloader)
        print(f"\nAfter unstructured – acc: {acc_after_unstruct:.2f}%")

        # ── Step 3: Fine-tune after pruning ─────────────────
        print(
            f"\n[STEP 3] Fine-tuning after pruning for {FINETUNE_EPOCHS_PRUNING} epochs..."
        )
        run = wandb.init(
            entity="scherpereelant-imt-atlantique",
            project="efficient deep learning Resnet depth comparison",
            config={
                "learning_rate": 0.001,
                "architecture": f"{model_str}_pruned_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}",
                "dataset": "CIFAR-10",
                "epochs": FINETUNE_EPOCHS_PRUNING,
                "batch_size": 64,
                "structured_ratio": STRUCTURED_RATIO,
                "unstructured_ratio": UNSTRUCTURED_RATIO,
                "k_bits_quantization": K_BITS,
                "stage": "pruning_finetune",
                "params_orig": params_orig,
                "params_after_struct": params_after_struct,
                "acc_orig": acc_orig,
                "acc_after_struct": acc_after_struct,
                "acc_after_unstruct": acc_after_unstruct,
            },
        )

        model = train_model(
            model, epochs=FINETUNE_EPOCHS_PRUNING, run=run, stage_name="pruning_ft"
        )

        # Make unstructured pruning permanent
        remove_pruning_masks(pruned_modules)

        acc_after_pruning_ft = evaluate(model, testloader)
        print(f"\nAccuracy after pruning fine-tune: {acc_after_pruning_ft:.2f}%")
        run.log({"pruning_ft/final_test_acc": acc_after_pruning_ft})
        run.finish()

        # ── Step 4: Apply fake quantization ─────────────────
        print(f"\n[STEP 4] Applying {K_BITS}-bit fake quantization...")
        model_quantized = apply_fake_quantization(model, k_bits=K_BITS, verbose=True)
        model_quantized = model_quantized.to(device)

        acc_after_quant = evaluate(model_quantized, testloader)
        print(f"Accuracy after quantization (before fine-tune): {acc_after_quant:.2f}%")

        # ── Step 5: Fine-tune after quantization ────────────
        print(
            f"\n[STEP 5] Fine-tuning after quantization for {FINETUNE_EPOCHS_QUANT} epochs..."
        )
        run = wandb.init(
            entity="scherpereelant-imt-atlantique",
            project="efficient deep learning Resnet depth comparison",
            config={
                "learning_rate": 0.0005,
                "architecture": f"{model_str}_pruned_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_quant{K_BITS}b",
                "dataset": "CIFAR-10",
                "epochs": FINETUNE_EPOCHS_QUANT,
                "batch_size": 64,
                "structured_ratio": STRUCTURED_RATIO,
                "unstructured_ratio": UNSTRUCTURED_RATIO,
                "k_bits_quantization": K_BITS,
                "stage": "quantization_finetune",
                "acc_after_pruning_ft": acc_after_pruning_ft,
                "acc_after_quant": acc_after_quant,
            },
        )

        model_quantized = train_model(
            model_quantized,
            epochs=FINETUNE_EPOCHS_QUANT,
            run=run,
            stage_name="quant_ft",
        )

        acc_final = evaluate(model_quantized, testloader)
        print(f"\nFinal accuracy after full pipeline: {acc_final:.2f}%")
        run.log({"quant_ft/final_test_acc": acc_final})
        run.finish()

        params_final = count_parameters(model_quantized)

        # ── Save ────────────────────────────────────────────
        torch.save(
            {
                "net": model_quantized.state_dict(),
                "params": params_final,
                "structured_ratio": STRUCTURED_RATIO,
                "unstructured_ratio": UNSTRUCTURED_RATIO,
                "k_bits": K_BITS,
                "architecture": f"{model_str}_struct{int(STRUCTURED_RATIO * 100)}_unstruct{int(UNSTRUCTURED_RATIO * 100)}_quant{K_BITS}b",
                "dataset": "CIFAR-10",
                "acc_orig": acc_orig,
                "acc_after_struct": acc_after_struct,
                "acc_after_unstruct": acc_after_unstruct,
                "acc_after_pruning_ft": acc_after_pruning_ft,
                "acc_after_quant": acc_after_quant,
                "acc_final": acc_final,
                "compression_ratio": params_orig / params_final,
            },
            CHECKPOINT_OUT,
        )

        print(f"\nFinal compression: {params_orig / params_final:.2f}x")
        print(f"Parameters: {params_final:,} (original: {params_orig:,})")
        print(f"Saved to {CHECKPOINT_OUT}")
        print(f"\n{'=' * 60}\n")
        print(f"Saved to {CHECKPOINT_OUT}")
        print(f"\n{'=' * 60}\n")

    wandb.finish()
