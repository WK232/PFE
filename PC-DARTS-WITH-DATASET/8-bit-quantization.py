import torch
import torch.nn as nn
import torch.quantization
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset
import utils
import genotypes
import os

# === CONFIGURATION ===
model_path = '/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/eval-EXP-20250617-095107/PC-DARTS.pth'
data_path = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset'
batch_size = 8
save_path = 'quantized_model.pth'
device = torch.device('cpu')  # Quantized models must run on CPU

print("🔧 Loading the trained FP32 model...")
genotype = genotypes.PCDARTS
model_fp32 = Network(C=36, num_classes=2, layers=20, auxiliary=False, genotype=genotype)
model_fp32.drop_path_prob = 0.0  # Disable drop path for quantization
model_fp32.load_state_dict(torch.load(model_path, map_location=device))
model_fp32.eval()
print("✅ Model loaded and set to eval mode.")

print("📐 Setting up quantization config...")
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(f"✅ QConfig set to: {model_fp32.qconfig}")

# (Optional) fuse modules if model supports it (example below, commented out)
# print("🔗 Fusing layers (if applicable)...")
# model_fp32 = torch.quantization.fuse_modules(model_fp32, [['conv', 'bn', 'relu']], inplace=True)
# print("✅ Fusion complete.")

print("🔍 Preparing the model for static quantization...")
torch.quantization.prepare(model_fp32, inplace=True)
print("✅ Model prepared.")

print("📊 Loading calibration dataset...")
dataset = CoherencePhaseSegmentationDataset(data_path)
calib_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print(f"✅ Loaded calibration data with {len(dataset)} samples.")

print("🧪 Starting calibration...")
with torch.no_grad():
    for idx, ((coh, pha), *_) in enumerate(calib_loader):
        input_tensor = torch.cat([coh, pha], dim=1)
        model_fp32(input_tensor)
        if idx % 10 == 0:
            print(f"  ➤ Calibration batch {idx + 1}/{len(calib_loader)}")
print("✅ Calibration complete.")

print("⚙️ Converting the model to 8-bit quantized version...")
model_int8 = torch.quantization.convert(model_fp32, inplace=False)
print("✅ Conversion done.")

print(f"💾 Saving quantized model to {save_path}...")
torch.save(model_int8.state_dict(), save_path)
print("✅ Quantized model saved successfully.")

# === Optional: Evaluate quantized model ===
def evaluate(model, data_loader):
    print("📈 Evaluating quantized model...")
    model.eval()
    acc_meter = utils.AvgrageMeter()
    with torch.no_grad():
        for (coh, pha), target in data_loader:
            input_tensor = torch.cat([coh, pha], dim=1)
            logits, _ = model(input_tensor)
            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)
            acc, _, _ = utils.pixel_metrics(logits, target, num_classes=2)
            acc_meter.update(acc, input_tensor.size(0))
    print(f"🎯 Quantized Model Accuracy: {acc_meter.avg:.4f}")

# Uncomment this line to run evaluation after quantization
evaluate(model_int8, calib_loader)
