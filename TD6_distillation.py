# =========================
# Imports
# =========================
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb
from pytorch_cifar.models.resnet import ResNet10, ResNet12, ResNet14, ResNet16, ResNet18

# =========================
# Config
# =========================
TEACHER_CKPT_PATH = "checkpoints/ResNet18_mixup_cos.pth"
STUDENT_CKPT_PATH = "checkpoints/ResNet10_mixup_cos_SGD.pth"
DISTILL_EPOCHS = 60
TEMPERATURE = 4.0
ALPHA = 0.7
LEARNING_RATE = 0.01
BATCH_SIZE = 64


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
        transforms.RandomCrop(32, padding=2),  # moins de padding
        transforms.RandomHorizontalFlip(p=0.3),  # moins de flips
        transforms.RandomRotation(2),  # rotation très faible
        transforms.RandomGrayscale(p=0.03),  # très peu de grayscale
        transforms.ColorJitter(
            brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01
        ),  # effet très léger
        transforms.RandomAffine(
            degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)
        ),  # translation/scale très faible
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

trainloader = DataLoader(c10train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(c10test, batch_size=BATCH_SIZE)


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


def _extract_resnet_depth(text):
    if not text:
        return None
    match = re.search(r"ResNet(10|12|14|16|18)", text)
    if match:
        return int(match.group(1))
    return None


def build_resnet_from_checkpoint(ckpt_path, checkpoint):
    architecture = checkpoint.get("architecture", "")
    depth = _extract_resnet_depth(architecture)
    if depth is None:
        depth = _extract_resnet_depth(os.path.basename(ckpt_path))
    if depth is None:
        raise ValueError(
            f"Impossible d'inférer l'architecture depuis {ckpt_path} (architecture='{architecture}')."
        )

    depth_to_model = {
        10: ResNet10,
        12: ResNet12,
        14: ResNet14,
        16: ResNet16,
        18: ResNet18,
    }
    return depth_to_model[depth](), f"ResNet{depth}"


def load_model_from_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model, model_name = build_resnet_from_checkpoint(ckpt_path, checkpoint)
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)
    return model, model_name, checkpoint


def train_student_with_distillation(
    student, teacher, epochs, temperature, alpha, run=None
):
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.SGD(
        student.parameters(),
        lr=run.config["learning_rate"] if run else LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

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
        teacher_test_acc = evaluate(teacher, testloader)
        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} - Loss: {avg_total_loss:.4f} "
            f"(CE: {avg_ce_loss:.4f}, KD: {avg_kd_loss:.4f}) - "
            f"Student Train: {student_train_acc:.2f}% - Student Test: {student_test_acc:.2f}%"
        )

        if run is not None:
            run.log(
                {
                    "epoch": epoch,
                    "distill/loss_total": avg_total_loss,
                    "distill/loss_ce": avg_ce_loss,
                    "distill/loss_kd": avg_kd_loss,
                    "student/train_acc": student_train_acc,
                    "student/test_acc": student_test_acc,
                    "teacher/test_acc": teacher_test_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    return student


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    teacher_model, teacher_name, teacher_ckpt = load_model_from_checkpoint(
        TEACHER_CKPT_PATH
    )
    student_model, student_name, student_ckpt = load_model_from_checkpoint(
        STUDENT_CKPT_PATH
    )
    print(
        f"Teacher: {teacher_name} ({count_parameters(teacher_model) / 1e6:.2f}M params), "
        f"Student: {student_name} ({count_parameters(student_model) / 1e6:.2f}M params)"
    )
    print(f"Teacher base acc: {evaluate(teacher_model, testloader):.2f}%")
    print(f"Student base acc: {evaluate(student_model, testloader):.2f}%")

    teacher_tag = os.path.splitext(os.path.basename(TEACHER_CKPT_PATH))[0]
    student_tag = os.path.splitext(os.path.basename(STUDENT_CKPT_PATH))[0]

    wandb.login()
    run = wandb.init(
        entity="scherpereelant-imt-atlantique",
        project="efficient deep learning distillation",
        config={
            "learning_rate": LEARNING_RATE,
            "teacher_checkpoint": TEACHER_CKPT_PATH,
            "student_checkpoint": STUDENT_CKPT_PATH,
            "teacher_architecture": teacher_ckpt.get("architecture", teacher_name),
            "student_architecture": student_ckpt.get("architecture", student_name),
            "dataset": "CIFAR-10",
            "epochs": DISTILL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "optimizer": "SGD",
        },
    )

    student_model = train_student_with_distillation(
        student=student_model,
        teacher=teacher_model,
        epochs=run.config["epochs"],
        temperature=run.config["temperature"],
        alpha=run.config["alpha"],
        run=run,
    )

    output_name = (
        f"{student_name}_distilled_from_{teacher_name}_T{run.config['temperature']}_"
        f"alpha{run.config['alpha']}_{run.config['epochs']}epochs_"
        f"{student_tag}_from_{teacher_tag}.pth"
    )
    output_path = os.path.join("checkpoints", output_name)

    state = {
        "net": student_model.state_dict(),
        "params": count_parameters(student_model),
        "epochs": run.config["epochs"],
        "architecture": f"{student_name} - Distilled from {teacher_name}",
        "dataset": "CIFAR-10",
        "teacher_checkpoint": TEACHER_CKPT_PATH,
        "student_checkpoint": STUDENT_CKPT_PATH,
        "temperature": run.config["temperature"],
        "alpha": run.config["alpha"],
    }
    torch.save(state, output_path)
    final_acc = evaluate(student_model, testloader)
    print(f"Model saved to {output_path}")
    print(f"Final student test accuracy after distillation: {final_acc:.2f}%")

    run.log(
        {
            "student/final_test_acc": final_acc,
            "student/params_million": count_parameters(student_model) / 1e6,
        }
    )

    artifact = wandb.Artifact(
        name=f"{student_name.lower()}_distilled_from_{teacher_name.lower()}_{student_tag}",
        type="model",
        metadata={
            "teacher_checkpoint": TEACHER_CKPT_PATH,
            "student_checkpoint": STUDENT_CKPT_PATH,
            "teacher_architecture": teacher_ckpt.get("architecture", teacher_name),
            "student_architecture": student_ckpt.get("architecture", student_name),
            "dataset": "CIFAR-10",
            "epochs": run.config["epochs"],
            "temperature": run.config["temperature"],
            "alpha": run.config["alpha"],
            "final_student_test_acc": final_acc,
        },
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

    run.finish()
    wandb.finish()
