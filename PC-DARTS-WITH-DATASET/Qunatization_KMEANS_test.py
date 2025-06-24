import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes
import utils

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser("Test FP16 Compressed Model")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='Path to dataset')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/fp16_compressed_model.pth', help='Path to FP16 model')
args = parser.parse_args()

# ----------------------------
# Load Model
# ----------------------------
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
model = Network(36, 2, 20, False, genotypes.PCDARTS)
model.drop_path_prob = 0.0  # Disable drop path for evaluation

model.to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print("param number = %d", utils.count_parameters(model))
print("✅ FP16 Model loaded.")

# ----------------------------
# Dataset
# ----------------------------
dataset = CoherencePhaseSegmentationDataset(args.data)
val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# ----------------------------
# Metrics
# ----------------------------
criterion = nn.CrossEntropyLoss()
acc_meter = utils.AvgrageMeter()
recall_meter = utils.AvgrageMeter()
loss_meter = utils.AvgrageMeter()

def compute_iou_single_class(pred_mask, target_mask, class_id=1):
    pred_inds = (pred_mask == class_id)
    target_inds = (target_mask == class_id)
    intersection = np.logical_and(pred_inds, target_inds).sum()
    union = np.logical_or(pred_inds, target_inds).sum()
    if union == 0:
        return None
    return intersection / union

# ----------------------------
# Visualization
# ----------------------------
def visualize_sample(coherence, phase, pred_mask, target_mask, iou, acc, idx=0):
    coherence = coherence.squeeze().cpu().numpy()
    phase = phase.squeeze().cpu().numpy()
    pred_mask = np.array(pred_mask)
    target_mask = np.array(target_mask)

    pred_img = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    target_img = np.zeros((*target_mask.shape, 3), dtype=np.uint8)

    colors = [(0, 0, 0), (255, 0, 0)]
    for i, color in enumerate(colors):
        pred_img[pred_mask == i] = color
        target_img[target_mask == i] = color

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 4, 1); plt.imshow(coherence, cmap='gray'); plt.title("Coherence"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(phase, cmap='gray'); plt.title("Phase"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(pred_img); plt.title("Prediction"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(target_img); plt.title("Ground Truth"); plt.axis('off')
    plt.suptitle(f"Sample {idx} - Acc: {acc:.2f} | IoU: {iou:.2f}")
    plt.tight_layout()
    plt.savefig(f"visual_sample_{idx}.png")
    plt.close()

# ----------------------------
# Evaluation Loop
# ----------------------------
iou_sum = 0.0
valid_iou_count = 0
sample_count = 0

with torch.no_grad():
    for step, ((coh, pha), target) in enumerate(val_loader):
        coh, pha, target = coh.to(device), pha.to(device), target.to(device)
        input_tensor = torch.cat([coh, pha], dim=1)

        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[0]

        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        loss = criterion(logits, target.long())
        acc, recall, _ = utils.pixel_metrics(logits, target, 2)
        n = input_tensor.size(0)

        acc_meter.update(acc, n)
        recall_meter.update(recall, n)
        loss_meter.update(loss.item(), n)

        preds = torch.argmax(logits, dim=1)

        for i in range(preds.size(0)):
            pred_np = preds[i].cpu().numpy()
            target_np = target[i].cpu().numpy()
            iou = compute_iou_single_class(pred_np, target_np, class_id=1)
            if iou is not None:
                iou_sum += iou
                valid_iou_count += 1

            # Save visualizations for 2 samples max
            if sample_count < 2:
                visualize_sample(coh[i], pha[i], pred_np, target_np, iou or 0, acc, idx=sample_count)
                sample_count += 1

# ----------------------------
# Results
# ----------------------------
avg_iou = iou_sum / valid_iou_count if valid_iou_count > 0 else 0.0

print("\n=== FP16 Model Evaluation ===")
print(f"Avg Accuracy: {acc_meter.avg:.4f}")
print(f"Avg Recall: {recall_meter.avg:.4f}")
print(f"Avg Loss: {loss_meter.avg:.4f}")
print(f"Avg IoU (Class 1): {avg_iou:.4f}")
