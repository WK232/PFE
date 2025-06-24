# ----------------------------
# PART 1: FP16 COMPRESSION SCRIPT
# ----------------------------
import torch
import torch.nn as nn
import argparse
from model import NetworkCIFAR as Network
import genotypes

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/eval-EXP-20250617-095107/PC-DARTS.pth')
parser.add_argument('--save_path', type=str, default='fp16_compressed_model.pth')
args = parser.parse_args()

# Load full precision model
model = Network(36, 2, 20, False, genotypes.PCDARTS)
state_dict = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Convert parameters to float16
fp16_state_dict = {}
for name, param in model.state_dict().items():
    if param.dtype == torch.float32:
        fp16_state_dict[name] = param.half()
    else:
        fp16_state_dict[name] = param  # Keep non-float params as-is

# Save compressed model
torch.save(fp16_state_dict, args.save_path)
print(f"\u2705 Model compressed to float16 and saved to {args.save_path}")
