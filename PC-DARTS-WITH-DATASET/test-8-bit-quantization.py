import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes
import utils

# === CONFIGURATION ===
model_path = '/home/kharratw/Documents/tessssst/PFE/quantized_model.pth'
data_path = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset'
batch_size = 8
is_quantized = True  # ⚠️ Set to True for quantized model
device = torch.device('cpu')  # Quantized models require CPU

# === Load model architecture ===
print("🧠 Instantiating model architecture...")
genotype = genotypes.PCDARTS  # or whatever genotype you're using
model = Network(C=16, num_classes=2, layers=8, auxiliary=False, genotype=genotype)

if is_quantized:
    print("🔧 Preparing model for quantization-aware evaluation...")
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
else:
    print("🧠 Loading full-precision model...")

# === Load model weights ===
print(f"📦 Loading weights from {model_path} ...")
state_dict = torch.load(model_path, map_location=device,weights_only=False)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

# === Load dataset ===
print("📊 Loading dataset...")
dataset = CoherencePhaseSegmentationDataset(data_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print(f"✅ Loaded dataset with {len(dataset)} samples.")

# === Evaluation ===
def evaluate(model, loader):
    model.eval()
    acc_meter = utils.AvgrageMeter()
    recall_meter = utils.AvgrageMeter()
    f1_meter = utils.AvgrageMeter()

    with torch.no_grad():
        for idx, ((coh, pha), target) in enumerate(loader):
            input_tensor = torch.cat([coh, pha], dim=1).to(device)
            target = target.to(device)

            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

            acc, recall, f1 = utils.pixel_metrics(logits, target, num_classes=2)

            acc_meter.update(acc, input_tensor.size(0))
            recall_meter.update(recall, input_tensor.size(0))
            f1_meter.update(f1, input_tensor.size(0))

            if idx % 10 == 0:
                print(f"  ➤ Batch {idx+1}/{len(loader)} | Acc: {acc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print("\n📈 Final Evaluation Metrics:")
    print(f"  🔹 Pixel Accuracy: {acc_meter.avg:.4f}")
    print(f"  🔹 Recall:         {recall_meter.avg:.4f}")
    print(f"  🔹 F1 Score:       {f1_meter.avg:.4f}")

evaluate(model, data_loader)
