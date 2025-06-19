import torch
import torch.nn as nn
import argparse
import genotypes
from model import Network  # adjust this import if needed
import os

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth', help='Path to the original model (.pth)')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--save_path', type=str, default='DARTS_8bit_quantized.pth', help='Where to save the quantized model')
parser.add_argument('--init_channels', type=int, default=36)
parser.add_argument('--layers', type=int, default=20)
parser.add_argument('--auxiliary', action='store_true', default=False)
args = parser.parse_args()


def quantize_tensor(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Avoid divide-by-zero
    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = round(qmin - min_val / scale)
        zero_point = max(qmin, min(qmax, zero_point))

    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return q_tensor, scale, zero_point


# Load full-precision model
model = Network(args.init_channels, 2, args.layers, args.auxiliary, getattr(genotypes, args.arch))
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

quantized_state = {}

# Quantize only Conv2d weights
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"Quantizing {name}...")

            weight = module.weight.data
            q_weight, scale, zero_point = quantize_tensor(weight)

            quantized_state[name + '.weight'] = {
                'q_weight': q_weight,
                'scale': torch.tensor(scale, dtype=torch.float32),
                'zero_point': torch.tensor(zero_point, dtype=torch.int)
            }

# Save the quantized weights and metadata
torch.save(quantized_state, args.save_path)
print(f"Quantized model saved to {args.save_path}")
