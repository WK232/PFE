import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from model import Network  # Your DARTS model definition
import genotypes
import argparse
# ---------------------------
# Settings
# ---------------------------

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDatasetclear', help='location of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/DARTS_pruned.pth')
args = parser.parse_args()

PRUNING_PERCENTAGE = 0.4  # Prune 40% of weights in each Conv layer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = '/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth'  # Path to trained model
SAVE_PATH = 'DARTS_pruned.pth'

# ---------------------------
# Load Pretrained Model
# ---------------------------
genotype = genotypes.DARTS  # Or your custom genotype
model = Network(args.init_channels, 2, args.layers, args.auxiliary, genotype)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Original model loaded.")

# ---------------------------
# Pruning Function
# ---------------------------
def prune_model_l1_unstructured(model, amount=0.4):
    """
    Apply L1 unstructured pruning to all Conv2d layers in the model
    and remove the masked (pruned) weights permanently.
    
    Args:
        model: The PyTorch model to prune.
        amount: The fraction of weights to prune in each Conv2d layer.
    
    Returns:
        The pruned model with weights permanently removed.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Pruning {name} - {amount * 100:.1f}% of weights")
            # Apply pruning
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Remove the pruning reparam (make it permanent)
            prune.remove(module, 'weight')
    
    return model

# ---------------------------
# Apply Pruning
# ---------------------------
#model = prune_model_l1_unstructured(model, amount=PRUNING_PERCENTAGE)

import torch

zeros = 0
total = 0
for name, param in model.named_parameters():
    if 'weight' in name:
        zeros += torch.sum(param == 0).item()
        total += param.numel()
print(f"Zeroed weights: {zeros}/{total} ({100 * zeros / total:.2f}%)")

# ---------------------------
# Save Pruned Model
# ---------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"Pruned model saved to {SAVE_PATH}")