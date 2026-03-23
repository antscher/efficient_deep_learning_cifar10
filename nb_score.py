import torch
from pytorch_cifar.models import *
from pytorch_cifar.models import resnet_fact
import matplotlib.pyplot as plt
import numpy as np

# Liste de tous les modèles disponibles dans le repo kuangliu/pytorch-cifar
model_dict = {
    "VGG16": lambda: VGG('VGG16'),
    "VGG19": lambda: VGG('VGG19'),
    "ResNet10": lambda: ResNet10(),
    "ResNet12": lambda: ResNet12(),
    "ResNet14": lambda: ResNet14(),
    "ResNet16": lambda: ResNet16(),
    "ResNet18": lambda: ResNet18(),
    "ResNet14_fact": lambda: ResNet14_fact(),
    "ResNet16_fact": lambda: ResNet16_fact(),
    "ResNet18_fact": lambda: ResNet18_fact(),
    "ResNet50": lambda: ResNet50(),
    "ResNet101": lambda: ResNet101(),
    "PreActResNet18": lambda: PreActResNet18(),
    "GoogLeNet": lambda: GoogLeNet(),
    "DenseNet121": lambda: DenseNet121(),
    "ResNeXt29_2x64d": lambda: ResNeXt29_2x64d(),
    "ResNeXt29_32x4d": lambda: ResNeXt29_32x4d(),
    "MobileNet": lambda: MobileNet(),
    "MobileNetV2": lambda: MobileNetV2(),
    "DPN92": lambda: DPN92(),
    "ShuffleNetG2": lambda: ShuffleNetG2(),
    "SENet18": lambda: SENet18(),
    "EfficientNetB0": lambda: EfficientNetB0(),
    "RegNetX_200MF": lambda: RegNetX_200MF(),
    "SimpleDLA": lambda: SimpleDLA(),
    "DLA": lambda: DLA()
}

# Données d'accuracy pour chaque modèle
accuracy_data = {
    "VGG16": 92.64,
    "ResNet18": 93.02,
    "ResNet14_fact": 0,
    "ResNet16_fact": 0,
    "ResNet18_fact": 0,
    "ResNet50": 93.62,
    "ResNet101": 93.75,
    "RegNetX_200MF": 94.24,
    "RegNetY_400MF": 94.29,
    "MobileNetV2": 94.43,
    "ResNeXt29_32x4d": 94.73,
    "ResNeXt29_2x64d": 94.82,
    "SimpleDLA": 94.89,
    "DenseNet121": 95.04,
    "PreActResNet18": 95.11,
    "DPN92": 95.16,
    "DLA": 95.47
}

# --- Ajout manuel de modèles personnalisés ---
# Format conservé pour compatibilité : { 'NomModel': [accuracy, nb_param (en millions)] }
custom_models = {
    # "ResNet50 - simple (b:64,e:50,lr:0.1)": [92.12, 23.52],
    # "ResNet50 - different DA": [91.14, 23.52],
    # "ResNet50 - Mixup": [91.49, 23.52],
    "ResNet18 - DA (b:64,e:70,lr:0.1)": [93.46, 11.17],
    "ResNet18 - Mixup": [93.16, 11.17],
}

print(f"{'Model Name':<30} | {'Parameters':<15} | {'Size (MB)':<10} | {'FLOPs':<12} | {'Score':<10}")
print("-" * 90)


def compute_flops(model, input_size=(1, 3, 32, 32)):
    hooks = []
    flops = {
        'total': 0
    }

    def conv_hook(module, inp, out):
        # out shape: (batch, out_c, out_h, out_w)
        batch_size = out.shape[0]
        out_c = out.shape[1]
        out_h = out.shape[2]
        out_w = out.shape[3]
        in_c = module.in_channels
        kernel_h, kernel_w = module.kernel_size
        groups = module.groups if hasattr(module, 'groups') else 1
        # multiply-add counts as 2 ops
        ops = batch_size * out_c * out_h * out_w * (in_c // groups) * kernel_h * kernel_w * 2
        flops['total'] += int(ops)

    def linear_hook(module, inp, out):
        batch_size = out.shape[0]
        in_features = module.in_features
        out_features = module.out_features
        ops = batch_size * in_features * out_features * 2
        flops['total'] += int(ops)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        try:
            inp = torch.zeros(input_size)
            _ = model(inp)
        except Exception:
            # If forward with dummy input fails, return 0 as fallback
            pass

    for h in hooks:
        h.remove()

    return flops['total']


def compute_score(w, f, p_s=0.0, p_u=0.0, q_w=32.0, q_a=32.0):
    # Formula from attachment:
    # score = ( [1 - (p_s + p_u)] * (q_w/32) * w ) / 5.6e6
    #       + ( (1 - p_s) * max(q_w, q_a)/32 * f ) / 2.8e8
    term1 = (1.0 - (p_s + p_u)) * (q_w / 32.0) * float(w) / 5.6e6
    term2 = (1.0 - p_s) * (max(q_w, q_a) / 32.0) * float(f) / 2.8e8
    return term1 + term2


# Stockage des données pour le plot (score au lieu de params)
score_list = []
accuracy_list = []
names_list = []

for name, model_fn in model_dict.items():
    try:
        net = model_fn()
        total_params = sum(p.numel() for p in net.parameters())
        size_mb = total_params * 4 / (1024 * 1024)
        # Estimate FLOPs using simple hooks (batch size 1, CIFAR input)
        flops = compute_flops(net, input_size=(1, 3, 32, 32))
        # Default compression/quantization parameters (can be adjusted later)
        score = compute_score(total_params, flops, p_s=0.0, p_u=0.0, q_w=32.0, q_a=32.0)

        print(f"{name:<30} | {total_params/1e6:>8.2f} M     | {size_mb:>8.2f} MB | {flops:>10,d} | {score:>8.4f}")

        if name in accuracy_data:
            score_list.append(score)
            accuracy_list.append(accuracy_data[name])
            names_list.append(name)

        del net

    except Exception as e:
        print(f"{name:<30} | ERROR: {str(e)}")

print("-" * 90)

# Ajout des modèles personnalisés au plot (estimation depuis nb_param donné)
if custom_models:
    for name, (acc, params_millions) in custom_models.items():
        w = params_millions * 1e6
        # fallback FLOPs estimate: proportional to params (rough heuristic)
        f_est = int(w * 2)
        score = compute_score(w, f_est, p_s=0.0, p_u=0.0, q_w=32.0, q_a=32.0)
        score_list.append(score)
        accuracy_list.append(acc)
        names_list.append(name)

    plt.figure(figsize=(12, 8))
    custom_names = set(custom_models.keys())
    x_custom = [score_list[i] for i, n in enumerate(names_list) if n in custom_names]
    y_custom = [accuracy_list[i] for i, n in enumerate(names_list) if n in custom_names]
    x_std = [score_list[i] for i, n in enumerate(names_list) if n not in custom_names]
    y_std = [accuracy_list[i] for i, n in enumerate(names_list) if n not in custom_names]
    names_std = [n for n in names_list if n not in custom_names]
    names_custom = [n for n in names_list if n in custom_names]

    plt.scatter(x_std, y_std, s=100, alpha=0.6, c='tab:blue', label='Other models')
    for i, name in enumerate(names_std):
        plt.annotate(name, (x_std[i], y_std[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    if x_custom:
        plt.scatter(x_custom, y_custom, s=200, alpha=0.9, c='red', edgecolor='black', label='Our models trained', zorder=5)
        for i, name in enumerate(names_custom):
            plt.annotate(name, (x_custom[i], y_custom[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, color='red', fontweight='bold', alpha=0.95)

    plt.xlabel('Score (see nb_score.py compute_score)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig('accuracy_vs_score.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as: accuracy_vs_score.png")
    plt.show()