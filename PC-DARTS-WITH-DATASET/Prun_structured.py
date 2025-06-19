import torch
import torch_pruning as tp
import argparse
import genotypes
from model import NetworkCIFAR as Network  # Assumes Network is used across architectures

# -------------------------------
# ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/eval-EXP-20250617-095107/PC-DARTS.pth', help='Path to the original model weights')
parser.add_argument('--save_path', type=str, default='model_pruned.pth', help='Path to save pruned model')
parser.add_argument('--prune_ratio', type=float, default=0.4, help='Channel pruning ratio')
parser.add_argument('--init_channels', type=int, default=36, help='Initial channels')
parser.add_argument('--layers', type=int, default=20, help='Number of layers')
parser.add_argument('--auxiliary', action='store_true', help='Use auxiliary head')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--arch', type=str, default='PCDARTS', help='Architecture name (e.g., DARTS, PCDARTS)')
args = parser.parse_args()

# -------------------------------
# LOAD MODEL & GENOTYPE
# -------------------------------
assert hasattr(genotypes, args.arch), f"❌ Architecture '{args.arch}' not found in genotypes!"
genotype = getattr(genotypes, args.arch)

print(f"✅ Using architecture: {args.arch}")
model = Network(
    args.init_channels,
    2,
    args.layers,
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
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        return output

wrapped_model = Wrapper(model)

# -------------------------------
# BUILD DEPENDENCY GRAPH
# -------------------------------
example_inputs = torch.randn(1, 2, 256, 256)  # Adjust channels if needed
DG = tp.DependencyGraph().build_dependency(wrapped_model, example_inputs)

# -------------------------------
# SELECT LAYERS TO IGNORE
# -------------------------------
ignored_layers = []
for name, module in wrapped_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        if module.groups == module.in_channels and module.groups > 1:
            ignored_layers.append(module)  # depthwise conv
        elif module.out_channels <= 8:
            ignored_layers.append(module)  # very small convs

print(f"🔍 Ignoring {len(ignored_layers)} layers during pruning")

# -------------------------------
# APPLY PRUNING
# -------------------------------
importance = tp.importance.MagnitudeImportance()
pruner = tp.pruner.MagnitudePruner(
    wrapped_model,
    example_inputs=example_inputs,
    importance=importance,
    iterative_steps=1,
    ch_sparsity=args.prune_ratio,
    ignored_layers=ignored_layers
)

pruner.step()
print("✅ Pruning completed successfully.")

# -------------------------------
# SAVE THE PRUNED MODEL
# -------------------------------
torch.save(wrapped_model.model, args.save_path)
print(f"✅ Saved pruned model to {args.save_path}")
