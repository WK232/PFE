import torch
import torch_pruning as tp
import argparse
import genotypes
from model import NetworkCIFAR as Network  # Assumes Network is used across architectures

# -------------------------------
# ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/kd-KD-EXP-20250625-163200-36channels-14layers/student_epoch_299.pth', help='Path to the original model weights')
parser.add_argument('--save_path', type=str, default='model_PCDARTS_14layers_36channels.pth', help='Path to save pruned model')
parser.add_argument('--prune_ratio', type=float, default=0.4, help='Channel pruning ratio')
parser.add_argument('--init_channels', type=int, default=36, help='Initial channels')
parser.add_argument('--layers', type=int, default=20, help='Number of layers')
parser.add_argument('--auxiliary', action='store_true', help='Use auxiliary head')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--arch', type=str, default='PCDARTS', help='Architecture name (e.g., DARTS, PCDARTS)')
args = parser.parse_args()

# -------------------------------
# LOAD MODEL & GENOTYPE
# -------------------------------
assert hasattr(genotypes, args.arch), f"❌ Architecture '{args.arch}' not found in genotypes!"
genotype = getattr(genotypes, args.arch)

print(f"✅ Using architecture: {args.arch}")
model = Network(
    36,
    2,
    14,
    args.auxiliary,
    genotype
)
model.drop_path_prob = args.drop_path_prob
# Load weights (support loading from .pth or state_dict)
state = torch.load(args.model_path, map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    model.load_state_dict(state['state_dict'])
else:
    model.load_state_dict(state)
model.eval()

# -------------------------------
# WRAP MODEL FOR PRUNING
# -------------------------------


# -------------------------------
# BUILD DEPENDENCY GRAPH
# -------------------------------


# -------------------------------
# SELECT LAYERS TO IGNORE
# -------------------------------


# -------------------------------
# APPLY PRUNING
# -------------------------------


# -------------------------------
# SAVE THE PRUNED MODEL
# -------------------------------
torch.save(model, args.save_path)
print(f"✅ Saved pruned model to {args.save_path}")
