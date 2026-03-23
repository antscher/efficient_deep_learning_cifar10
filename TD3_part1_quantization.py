# =========================
# Imports
# =========================
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import wandb
from pytorch_cifar.models.resnet import ResNet14

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
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(p=0.3),
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


def evaluate(model, loader, eval_device=device, half=False):
    correct, total = 0, 0
    model = model.to(eval_device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(eval_device), y.to(eval_device)
            if half:
                x = x.half()
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.0 * correct / total


def train_model(
    model,
    epochs,
    loader,
    test_loader,
    optimizer,
    run=None,
    stage_name="train",
    train_device=device,
    half=False,
):
    model = model.to(train_device)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(train_device), y.to(train_device)
            if half:
                x = x.half()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        acc_train = evaluate(model, loader, eval_device=train_device, half=half)
        acc_test = evaluate(model, test_loader, eval_device=train_device, half=half)
        scheduler.step()
        print(
            f"[{stage_name}] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc_train:.2f}% - Test Acc: {acc_test:.2f}%"
        )
        if run is not None:
            run.log(
                {
                    f"{stage_name}/epoch": epoch + 1,
                    f"{stage_name}/loss": avg_loss,
                    f"{stage_name}/train_acc": acc_train,
                    f"{stage_name}/test_acc": acc_test,
                    f"{stage_name}/lr": optimizer.param_groups[0]["lr"],
                }
            )

    return model


def apply_fake_4bit_quantization(model):
    model_quantized = copy.deepcopy(model).to(device)
    model_quantized.eval()

    fake_quant = FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=0,
        quant_max=15,
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

    return model_quantized


# We load the dictionary
loaded_cpt = torch.load("checkpoints/ResNet14_mixup_cos_SGD.pth", map_location=device)


# Define the model
model = ResNet14()

# Finally we can load the state_dict in order to load the trained parameters
model.load_state_dict(loaded_cpt["net"])
model = model.to(device)


model.eval()


# If you use this model for inference (= no further training), you need to set it into eval mode
print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
print(f"arch: {loaded_cpt['architecture']}")
print(f"Evaluation accuracy: {evaluate(model, testloader, half=False):.2f}%")

wandb.login()

epochs_after_quant = 30
lr = 0.001

print("Quantization in 16 bits (FP16)...")
run = wandb.init(
    entity="scherpereelant-imt-atlantique",
    project="efficient deep learning Resnet depth comparison",
    config={
        "learning_rate": 0.01,
        "architecture": "ResNet14-mixup-cos-16bits",
        "dataset": "CIFAR-10",
        "epochs_after_quantization": 30,
        "batch_size": 64,
        "optimizer": "SGD",
    },
)
model_fp16 = copy.deepcopy(model).half()
optimizer_fp16 = optim.SGD(
    model_fp16.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
)
model_fp16 = train_model(
    model_fp16,
    epochs=epochs_after_quant,
    loader=trainloader,
    test_loader=testloader,
    optimizer=optimizer_fp16,
    run=run,
    stage_name="fp16_finetune",
    train_device=device,
    half=True,
)
state_fp16 = {
    "net": model_fp16.state_dict(),
    "params": sum(p.numel() for p in model_fp16.parameters()),
    "epochs": epochs_after_quant,
    "architecture": "ResNet14",
    "dataset": "CIFAR-10",
    "quantization": "fp16",
}
torch.save(state_fp16, "checkpoints/ResNet14_mixup_cos_SGD_16b.pth")

print("Quantization in 8 bits (INT8 QAT)...")
run = wandb.init(
    entity="scherpereelant-imt-atlantique",
    project="efficient deep learning Resnet depth comparison",
    config={
        "learning_rate": 0.01,
        "architecture": "ResNet14-mixup-cos-8bits",
        "dataset": "CIFAR-10",
        "epochs_after_quantization": 30,
        "batch_size": 64,
        "optimizer": "SGD",
    },
)
model_int8_qat = copy.deepcopy(model).to(device)
model_int8_qat.train()
model_int8_qat.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
torch.ao.quantization.prepare_qat(model_int8_qat, inplace=True)
optimizer_int8_qat = optim.SGD(
    model_int8_qat.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
)
model_int8_qat = train_model(
    model_int8_qat,
    epochs=epochs_after_quant,
    loader=trainloader,
    test_loader=testloader,
    optimizer=optimizer_int8_qat,
    run=run,
    stage_name="int8_qat_finetune",
    train_device=device,
    half=False,
)
state_int8_qat = {
    "net": model_int8_qat.state_dict(),
    "params": sum(p.numel() for p in model_int8_qat.parameters()),
    "epochs": epochs_after_quant,
    "architecture": "ResNet14",
    "dataset": "CIFAR-10",
    "quantization": "int8_qat",
}
torch.save(state_int8_qat, "checkpoints/ResNet14_mixup_cos_SGD_8b.pth")

print("Quantization in 4 bits (FakeQuantize)...")
run = wandb.init(
    entity="scherpereelant-imt-atlantique",
    project="efficient deep learning Resnet depth comparison",
    config={
        "learning_rate": 0.01,
        "architecture": "ResNet14-mixup-cos-4bits",
        "dataset": "CIFAR-10",
        "epochs_after_quantization": 30,
        "batch_size": 64,
        "optimizer": "SGD",
    },
)
model_fake4 = apply_fake_4bit_quantization(model)
optimizer_fake4 = optim.SGD(
    model_fake4.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
)
model_fake4 = train_model(
    model_fake4,
    epochs=epochs_after_quant,
    loader=trainloader,
    test_loader=testloader,
    optimizer=optimizer_fake4,
    run=run,
    stage_name="fake4_finetune",
    train_device=device,
    half=False,
)
state_fake4 = {
    "net": model_fake4.state_dict(),
    "params": sum(p.numel() for p in model_fake4.parameters()),
    "epochs": epochs_after_quant,
    "architecture": "ResNet14",
    "dataset": "CIFAR-10",
    "quantization": "fake4",
}
torch.save(state_fake4, "checkpoints/ResNet14_mixup_cos_SGD_fake4.pth")

run.finish()
wandb.finish()
