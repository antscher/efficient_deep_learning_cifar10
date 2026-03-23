import torch
from pytorch_cifar.models import *
import matplotlib.pyplot as plt
import numpy as np

# Liste de tous les modèles disponibles dans le repo kuangliu/pytorch-cifar
# J'ai adapté les appels pour correspondre aux classes du dossier 'models'
model_dict = {
    "VGG16": lambda: VGG('VGG16'),
    "VGG19": lambda: VGG('VGG19'),
    "ResNet18": lambda: ResNet18(),
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
# Format : { 'NomModel': [accuracy, nb_param (en millions)] }
custom_models = {
    "ResNet50 - simple (b:64,e:50,lr:0.1)": [92.12, 23.52],
    "ResNet50 - different DA": [91.14, 23.52],
    "ResNet50 - Mixup": [91.49, 23.52],
    "ResNet18 - DA (b:64,e:70,lr:0.1)": [93.46, 11.17],
    "ResNet18 - Mixup": [93.16, 11.17],

    # Exemple : 'MonModel1': [94.1, 12],
}

print(f"{'Model Name':<20} | {'Parameters':<15} | {'Size (MB)':<10}")
print("-" * 50)

# Stockage des données pour le plot
params_list = []
accuracy_list = []
names_list = []

for name, model_fn in model_dict.items():
    try:
        # Instanciation du modèle
        net = model_fn()
        
        # Calcul du nombre de paramètres
        total_params = sum(p.numel() for p in net.parameters())
        
        # Estimation de la taille en mémoire (float32 = 4 bytes)
        size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"{name:<20} | {total_params:,.0f}".replace(",", " ") + f" | {size_mb:.2f} MB")
        
        # Stocker les données si l'accuracy est disponible
        if name in accuracy_data:
            params_list.append(total_params / 1e6)  # Convertir en millions
            accuracy_list.append(accuracy_data[name])
            names_list.append(name)
        
        # Nettoyage mémoire pour éviter de surcharger si exécuté sur GPU (bien que par défaut sur CPU ici)
        del net
        
    except Exception as e:
        print(f"{name:<20} | ERROR: {str(e)}")

print("-" * 50)

# Ajout des modèles personnalisés au plot
if custom_models:
    for name, (acc, params) in custom_models.items():
        params_list.append(params)
        accuracy_list.append(acc)
        names_list.append(name)

    # Plot all models
    plt.figure(figsize=(12, 8))
    # Séparer les modèles custom pour les mettre en valeur
    custom_names = set(custom_models.keys())
    x_custom = [params_list[i] for i, n in enumerate(names_list) if n in custom_names]
    y_custom = [accuracy_list[i] for i, n in enumerate(names_list) if n in custom_names]
    x_std = [params_list[i] for i, n in enumerate(names_list) if n not in custom_names]
    y_std = [accuracy_list[i] for i, n in enumerate(names_list) if n not in custom_names]
    names_std = [n for n in names_list if n not in custom_names]
    names_custom = [n for n in names_list if n in custom_names]

    # Modèles standards
    plt.scatter(x_std, y_std, s=100, alpha=0.6, c='tab:blue', label='Other models')
    for i, name in enumerate(names_std):
        plt.annotate(name, (x_std[i], y_std[i]), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    # Modèles custom en rouge
    if x_custom:
        plt.scatter(x_custom, y_custom, s=200, alpha=0.9, c='red', edgecolor='black', label='Our models trained', zorder=5)
        for i, name in enumerate(names_custom):
            plt.annotate(name, (x_custom[i], y_custom[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, color='red', fontweight='bold', alpha=0.95)

    plt.xlabel('Number of parameters (millions)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Number of parameters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig('accuracy_vs_params.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as: accuracy_vs_params.png")
    plt.show()