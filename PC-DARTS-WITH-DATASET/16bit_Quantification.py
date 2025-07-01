# ----------------------------
# PART 1: FP16 COMPRESSION SCRIPT
# ----------------------------
import torch
import torch.nn as nn
import argparse
from model import NetworkCIFAR as Network
import genotypes

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/model_PCDARTS_base.pth')
parser.add_argument('--save_path', type=str, default='fp16_compressed_model_basic.pth')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
args = parser.parse_args()

# Load full precision model
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
model = torch.load(args.model_path, map_location=device, weights_only=False)
model.to(device)

model = model.half()
torch.save(model, args.save_path)
print(f"\u2705 Model compressed to float16 and saved to {args.save_path}")
