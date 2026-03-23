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
from torch.ao.quantization import FakeQuantize, get_default_qat_qconfig_mapping
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb
from pytorch_cifar.models.resnet import ResNet10, ResNet12, ResNet14, ResNet16, ResNet18
from pytorch_cifar.models.resnet_fact import ResNet16_fact

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
    return 100.0 * correct / total


# ============================================================
# STRUCTURED PRUNING
# ============================================================


def prune_depthwise_conv(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new depthwise Conv2d keeping only the channels at keep_idx."""
    n_keep = len(keep_idx)
    new_conv = nn.Conv2d(
        in_channels=n_keep,
        out_channels=n_keep,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=n_keep,  # <--- CRITICAL: update groups to match new channels
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep_idx])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep_idx])
    return new_conv


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
        conv.in_channels,
        len(keep_idx),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[keep_idx])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias[keep_idx])
    return new_conv


def prune_conv_in_channels(conv: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    """Create a new Conv2d keeping only the input channels at keep_idx."""
    new_conv = nn.Conv2d(
        len(keep_idx),
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight[:, keep_idx])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def prune_batchnorm(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    """Create a new BatchNorm2d keeping only the channels at keep_idx."""
    new_bn = nn.BatchNorm2d(
        len(keep_idx),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight[keep_idx])
            new_bn.bias.copy_(bn.bias[keep_idx])
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean[keep_idx])
            new_bn.running_var.copy_(bn.running_var[keep_idx])
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def apply_structured_pruning(model, prune_ratio=0.3, verbose=True):
    """Structured filter pruning on FactorizedBasicBlocks."""
    if verbose:
        print(f"\n── Structured pruning (ratio={prune_ratio}) ──")

    # Adapt this if your model uses different naming (e.g., layer1, layer2, etc.)
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            # Only prune if it's our target block type
            if not hasattr(block, "conv1_pw"):
                continue

            # We prune the intermediate channels 'planes'.
            # These are generated by conv1_pw and consumed by conv2_pw.
            conv1_pw = block.conv1_pw
            n_total = conv1_pw.out_channels
            n_keep = max(1, int(round(n_total * (1.0 - prune_ratio))))

            # 1. Calculate norms on the layer that GENERATES the channels we want to prune
            norms = l1_norm_filter(conv1_pw)
            keep_idx = torch.argsort(norms, descending=True)[:n_keep]
            keep_idx, _ = keep_idx.sort()  # Sort to maintain original channel order

            if verbose:
                print(
                    f"  Internal channels: {n_total}→{n_keep} "
                    f"(removed {n_total - n_keep}, {(n_total - n_keep) / n_total * 100:.1f}%)"
                )

            # 2. Apply pruning along the forward pass sequence

            # conv1_pw output gets pruned
            block.conv1_pw = prune_conv_out_channels(conv1_pw, keep_idx)
            block.bn1_pw = prune_batchnorm(block.bn1_pw, keep_idx)

            # conv2_dw is depthwise, both in, out, and groups are pruned
            block.conv2_dw = prune_depthwise_conv(block.conv2_dw, keep_idx)
            block.bn2_dw = prune_batchnorm(block.bn2_dw, keep_idx)

            # conv2_pw input gets pruned (it must accept the reduced channels from conv2_dw)
            block.conv2_pw = prune_conv_in_channels(block.conv2_pw, keep_idx)

            # Notice we DO NOT touch conv1_dw, bn1_dw, or bn2_pw.
            # Doing so would alter the block's input/output size and break the residual shortcut.

    return model


# ============================================================
# UNSTRUCTURED PRUNING
# ============================================================


def apply_unstructured_pruning(model, prune_ratio=0.7, verbose=True):
    """Global unstructured L1 pruning on all Conv2d and Linear layers."""
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
    """Make unstructured pruning permanent."""
    for module, param in modules_to_prune:
        prune.remove(module, param)


# ============================================================
# QUANTIZATION
# ============================================================
def apply_fake_quantization(model):
    """
    Apply native PyTorch 8-bit fake quantization for QAT.
    """
    # 1. On charge la configuration standard de PyTorch pour le 8-bits
    input = (torch.randn(1, 3, 32, 32),)
    qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")

    # 2. Le modèle doit être en mode entraînement
    model.train()

    # 3. PyTorch scanne le modèle et insère automatiquement tous les "FakeQuantize"
    # sur les poids et les activations appropriés.
    student_model_quant = prepare_qat_fx(model, qconfig_mapping, input)

    return student_model_quant


# ============================================================
# DISTILLATION-GUIDED FINE-TUNING
# ============================================================


def train_with_distillation(
    student,
    teacher,
    epochs,
    temperature=4.0,
    alpha=0.7,
    learning_rate=0.001,
    run=None,
    stage_name="finetune",
):
    """
    Train student with knowledge distillation from teacher.

    Loss = alpha * KD_Loss + (1-alpha) * CE_Loss
    """
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    optimizer = optim.SGD(
        student.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"\nTraining with distillation (T={temperature}, α={alpha})...")

    for epoch in range(1, epochs + 1):
        student.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            loss_ce = ce_loss(student_logits, y)
            student_log_probs_t = torch.log_softmax(student_logits / temperature, dim=1)
            teacher_probs_t = torch.softmax(teacher_logits / temperature, dim=1)
            loss_kd = kd_loss(student_log_probs_t, teacher_probs_t) * (temperature**2)
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce

            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_ce_loss += loss_ce.item()
            running_kd_loss += loss_kd.item()

        avg_total_loss = running_total_loss / len(trainloader)
        avg_ce_loss = running_ce_loss / len(trainloader)
        avg_kd_loss = running_kd_loss / len(trainloader)
        student_train_acc = evaluate(student, trainloader)
        student_test_acc = evaluate(student, testloader)
        scheduler.step()

        print(
            f"  [{stage_name} {epoch}/{epochs}] Loss: {avg_total_loss:.4f} "
            f"(CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f}) - "
            f"Train: {student_train_acc:.2f}% - Test: {student_test_acc:.2f}%"
        )

        if run is not None:
            run.log(
                {
                    f"{stage_name}/epoch": epoch,
                    f"{stage_name}/loss_total": avg_total_loss,
                    f"{stage_name}/loss_ce": avg_ce_loss,
                    f"{stage_name}/loss_kd": avg_kd_loss,
                    f"{stage_name}/train_acc": student_train_acc,
                    f"{stage_name}/test_acc": student_test_acc,
                    f"{stage_name}/lr": optimizer.param_groups[0]["lr"],
                }
            )

    return student


# ============================================================
# Main Pipeline
# ============================================================


def pipeline_pruning_quantization_distill(
    teacher_model,
    student_model,
    teacher_name="Teacher",
    student_name="Student",
    structured_ratio=0.90,
    unstructured_ratio=0.5,
    k_bits=8,
    temperature=4.0,
    alpha=0.7,
    prune_finetune_epochs=25,
    quant_finetune_epochs=25,
    verbose=True,
):
    """
    Pipeline: Structured Pruning → Unstructured Pruning →
              Fine-tune with Distillation → Quantization →
              Fine-tune with Distillation

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Pre-trained student model
        teacher_name: Name for logging
        student_name: Name for logging
        structured_ratio: Fraction of filters to prune
        unstructured_ratio: Fraction of weights to prune
        k_bits: Quantization bit-width
        temperature: Distillation temperature
        alpha: KD loss weight (alpha * KD + (1-alpha) * CE)
        prune_finetune_epochs: Epochs for pruning fine-tune
        quant_finetune_epochs: Epochs for quantization fine-tune
        verbose: Print debug info

    Returns:
        dict: Results with models and metrics
    """

    # Move models to device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # Get baseline metrics
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    acc_teacher = evaluate(teacher_model, testloader)
    acc_student_base = evaluate(student_model, testloader)

    if verbose:
        print(f"\n{'=' * 70}")
        print("TD7: Pruning + Quantization with Distillation-Guided Fine-tuning")
        print(f"{'=' * 70}")
        print(
            f"Teacher: {teacher_name} ({teacher_params:,} params) - Acc: {acc_teacher:.2f}%"
        )
        print(
            f"Student: {student_name} ({student_params:,} params) - Acc: {acc_student_base:.2f}%"
        )

    wandb.login()
    run = wandb.init(
        entity="scherpereelant-imt-atlantique",
        project="efficient deep learning TD7",
        config={
            "pipeline": "pruning_quantization_with_distillation",
            "teacher": teacher_name,
            "student": student_name,
            "structured_ratio": structured_ratio,
            "unstructured_ratio": unstructured_ratio,
            "k_bits": k_bits,
            "temperature": temperature,
            "alpha": alpha,
            "prune_finetune_epochs": prune_finetune_epochs,
            "quant_finetune_epochs": quant_finetune_epochs,
        },
    )

    # ── Step 1: Structured Pruning ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[STEP 1] Structured Pruning")
    print(f"{'=' * 70}")

    student_model = apply_structured_pruning(
        student_model, prune_ratio=structured_ratio, verbose=verbose
    )
    student_model = student_model.to(device)

    student_params_after_struct = count_parameters(student_model)
    acc_after_struct = evaluate(student_model, testloader)

    if verbose:
        print(f"\nAfter structured pruning:")
        print(
            f"  Params: {student_params_after_struct:,} ({student_params / student_params_after_struct:.2f}x)"
        )
        print(f"  Accuracy: {acc_after_struct:.2f}%")

    # ── Step 2: Unstructured Pruning ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[STEP 2] Unstructured Pruning")
    print(f"{'=' * 70}")

    pruned_modules = apply_unstructured_pruning(
        student_model, prune_ratio=unstructured_ratio, verbose=verbose
    )

    acc_after_unstruct = evaluate(student_model, testloader)
    if verbose:
        print(f"\nAfter unstructured pruning:")
        print(f"  Accuracy: {acc_after_unstruct:.2f}%")

    # ── Step 3: Fine-tune with Distillation after Pruning ──────────────
    print(f"\n{'=' * 70}")
    print(
        f"[STEP 3] Fine-tune after Pruning with Distillation ({prune_finetune_epochs} epochs)"
    )
    print(f"{'=' * 70}")

    student_model = train_with_distillation(
        student=student_model,
        teacher=teacher_model,
        epochs=prune_finetune_epochs,
        temperature=temperature,
        alpha=alpha,
        learning_rate=0.001,
        run=run,
        stage_name="prune_distill_ft",
    )

    remove_pruning_masks(pruned_modules)
    acc_after_prune_ft = evaluate(student_model, testloader)

    if verbose:
        print(f"\nAccuracy after pruning fine-tune: {acc_after_prune_ft:.2f}%")
    run.log({"prune_distill_ft/final_acc": acc_after_prune_ft})

    # ── Step 4: Apply Fake Quantization ───────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[STEP 4] Apply {k_bits}-bit Fake Quantization")
    print(f"{'=' * 70}")

    student_model_quant = apply_fake_quantization(student_model)
    student_model_quant = student_model_quant.to(device)

    acc_after_quant = evaluate(student_model_quant, testloader)
    if verbose:
        print(
            f"\nAccuracy after quantization (before fine-tune): {acc_after_quant:.2f}%"
        )

    # ── Step 5: Fine-tune with Distillation after Quantization ────────
    print(f"\n{'=' * 70}")
    print(
        f"[STEP 5] Fine-tune after Quantization with Distillation ({quant_finetune_epochs} epochs)"
    )
    print(f"{'=' * 70}")

    student_model_quant = train_with_distillation(
        student=student_model_quant,
        teacher=teacher_model,
        epochs=quant_finetune_epochs,
        temperature=temperature,
        alpha=alpha,
        learning_rate=0.0005,
        run=run,
        stage_name="quant_distill_ft",
    )

    acc_final = evaluate(student_model_quant, testloader)
    student_params_final = count_parameters(student_model_quant)

    if verbose:
        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Final Accuracy: {acc_final:.2f}%")
        print(f"Final Params: {student_params_final:,}")
        print(f"Total Compression: {student_params / student_params_final:.2f}x")
        print(f"Accuracy Retention: {(acc_final / acc_student_base) * 100:.1f}%")
        print(f"vs Teacher: {acc_final:.2f}% vs {acc_teacher:.2f}%")
        print(f"{'=' * 70}\n")

    run.log(
        {
            "quant_distill_ft/final_acc": acc_final,
            "final/compression": student_params / student_params_final,
            "final/accuracy_retention": (acc_final / acc_student_base) * 100,
        }
    )
    run.finish()

    return {
        "student_model": student_model_quant,
        "teacher_model": teacher_model,
        "acc_teacher": acc_teacher,
        "acc_student_base": acc_student_base,
        "acc_after_struct": acc_after_struct,
        "acc_after_unstruct": acc_after_unstruct,
        "acc_after_prune_ft": acc_after_prune_ft,
        "acc_after_quant": acc_after_quant,
        "acc_final": acc_final,
        "params_student_base": student_params,
        "params_final": student_params_final,
        "compression_ratio": student_params / student_params_final,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Define models directly
    teacher = ResNet18()
    student = ResNet16_fact()

    # Load pre-trained weights
    teacher_ckpt = torch.load(
        "checkpoints/ResNet18_mixup_cos_SGD.pth", map_location=device
    )
    student_ckpt = torch.load(
        "checkpoints/ResNet16_mixup_cos_SGD_fact.pth", map_location=device
    )

    teacher.load_state_dict(teacher_ckpt["net"])
    student.load_state_dict(student_ckpt["net"])

    # Configuration
    STRUCTURED_RATIO = 0.82
    UNSTRUCTURED_RATIO = 0.3
    K_BITS = 8
    TEMPERATURE = 4.0
    ALPHA = 0.8
    PRUNE_FINETUNE_EPOCHS = 25
    QUANT_FINETUNE_EPOCHS = 25

    # Run pipeline
    results = pipeline_pruning_quantization_distill(
        teacher_model=teacher,
        student_model=student,
        teacher_name="ResNet16",
        student_name="ResNet14_fact",
        structured_ratio=STRUCTURED_RATIO,
        unstructured_ratio=UNSTRUCTURED_RATIO,
        k_bits=K_BITS,
        temperature=TEMPERATURE,
        alpha=ALPHA,
        prune_finetune_epochs=PRUNE_FINETUNE_EPOCHS,
        quant_finetune_epochs=QUANT_FINETUNE_EPOCHS,
        verbose=True,
    )

    # Save model
    output_path = f"checkpoints/TD7_ResNet10_s{int(STRUCTURED_RATIO * 100)}_u{int(UNSTRUCTURED_RATIO * 100)}_q{K_BITS}b.pth"
    torch.save(
        {
            "net": results["student_model"].state_dict(),
            "params": results["params_final"],
            "architecture": "ResNet10 - Pruned + Quantized + Distilled",
            "dataset": "CIFAR-10",
            "pipeline": "TD7",
            "teacher": "ResNet18",
            "student": "ResNet10",
            "structured_ratio": STRUCTURED_RATIO,
            "unstructured_ratio": UNSTRUCTURED_RATIO,
            "k_bits": K_BITS,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "acc_teacher": results["acc_teacher"],
            "acc_student_base": results["acc_student_base"],
            "acc_final": results["acc_final"],
            "compression_ratio": results["compression_ratio"],
        },
        output_path,
    )

    print(f"\nModel saved to: {output_path}")
    wandb.finish()
