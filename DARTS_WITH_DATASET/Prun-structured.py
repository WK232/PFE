import torch
import torch_pruning as tp
from model import Network
import genotypes
import argparse

# -------------------------------
# ARGUMENTS
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth')
parser.add_argument('--save_path', type=str, default='model_pruned.pth')
parser.add_argument('--prune_ratio', type=float, default=0.4)
parser.add_argument('--init_channels', type=int, default=36)
parser.add_argument('--layers', type=int, default=20)
parser.add_argument('--auxiliary', action='store_true')
parser.add_argument('--arch', type=str, default='DARTS')
args = parser.parse_args()

# -------------------------------
# LOAD MODEL & GENOTYPE
# -------------------------------
genotype = getattr(genotypes, args.arch)
model = Network(args.init_channels, 2, args.layers, args.auxiliary, genotype)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
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
example_inputs = torch.randn(1, 2, 256, 256)  # Adjust input shape if needed
DG = tp.DependencyGraph().build_dependency(wrapped_model, example_inputs)

# -------------------------------
# MANUAL IGNORED LAYER SELECTION
# -------------------------------
ignored_layers = []
for name, module in wrapped_model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        if module.groups == module.in_channels and module.groups > 1:
            ignored_layers.append(module)  # Skip depthwise conv
        elif module.out_channels <= 8:
            ignored_layers.append(module)  # Skip small conv layers

# -------------------------------
# PRUNING SETUP
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

# -------------------------------
# APPLY PRUNING
# -------------------------------
pruner.step()
print("✅ Pruning completed")

# -------------------------------
# SAVE PRUNED MODEL (ONLY CORE MODEL)
# -------------------------------
torch.save(wrapped_model.model, args.save_path)
print(f"✅ Saved pruned model to {args.save_path}")
