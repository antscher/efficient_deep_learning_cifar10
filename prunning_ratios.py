# function to run a python script and capture its output

from TD4_pruning_unstructured import apply_global_unstructured_pruning
from pytorch_cifar.models.resnet import ResNet18
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    for prune_ratio in [0.5, 0.75, 0.9, 0.95, 0.99]:
        loaded_cpt = torch.load('checkpoints/ResNet18_mixup_cos.pth',map_location=device)
        # Define the model
        model = ResNet18()
        #    Finally we can load the state_dict in order to load the trained parameters
        model.load_state_dict(loaded_cpt['net'])
        model = model.to(device)
        print(f"Applying global unstructured pruning with ratio {prune_ratio}...")
        apply_global_unstructured_pruning(model, prune_ratio=prune_ratio)