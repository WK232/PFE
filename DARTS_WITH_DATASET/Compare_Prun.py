import torch
import os
import time
import numpy as np
from model import Network
from torch.utils.data import DataLoader
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes


def calculate_accuracy(preds, target):
    correct = (preds == target).sum().item()
    total = target.numel()
    return correct / total

def compute_iou(pred_mask, target_mask, num_classes=2):
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (target_mask == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        iou = intersection / union if union != 0 else float('nan')
        iou_per_class.append(iou)

    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else float('nan')
    return mean_iou, iou_per_class

def evaluate_model(model, test_loader, device, num_classes=2):
    model.eval()
    total_accuracy = 0
    total_iou = 0
    total_samples = 0
    total_time = 0

    with torch.no_grad():
        for (coh, pha), target in test_loader:
            coh, pha, target = coh.to(device), pha.to(device), target.to(device)
            input = torch.cat([coh, pha], dim=1)

            start_time = time.time()
            output = model(input)
            end_time = time.time()
            total_time += (end_time - start_time)

            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)

            for i in range(preds.size(0)):
                acc = calculate_accuracy(preds[i], target[i])
                iou, _ = compute_iou(preds[i].cpu().numpy(), target[i].cpu().numpy(), num_classes)
                total_accuracy += acc
                total_iou += iou
                total_samples += 1

    avg_acc = total_accuracy / total_samples
    avg_iou = total_iou / total_samples
    avg_time = total_time / total_samples
    return avg_acc, avg_iou, avg_time

def model_stats(model_path):
    size_mb = os.path.getsize(model_path) / 1e6
    checkpoint = torch.load(model_path, map_location='cpu')
    model = Network(36, 2, 20, False, getattr(genotypes, "DARTS"))
    model.load_state_dict(checkpoint)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return size_mb, param_count

def compare_models(original_path, pruned_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    genotype = getattr(genotypes, "DARTS")

    test_dataset = CoherencePhaseSegmentationDataset(data_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    def load_model(path):
        model = Network(36, 2, 20, False, genotype)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
        return model.to(device)

    print("\nEvaluating Original Model...")
    orig_model = load_model(original_path)
    orig_acc, orig_iou, orig_time = evaluate_model(orig_model, test_loader, device)
    orig_size, orig_params = model_stats(original_path)

    print("\nEvaluating Pruned Model...")
    pruned_model = load_model(pruned_path)
    pruned_acc, pruned_iou, pruned_time = evaluate_model(pruned_model, test_loader, device)
    pruned_size, pruned_params = model_stats(pruned_path)

    print("\n=== Model Comparison Summary ===")
    print(f"{'Metric':<25}{'Original':<15}{'Pruned':<15}")
    print(f"{'Accuracy':<25}{orig_acc:.4f}{pruned_acc:>15.4f}")
    print(f"{'Mean IoU':<25}{orig_iou:.4f}{pruned_iou:>15.4f}")
    print(f"{'Model Size (MB)':<25}{orig_size:.2f}{pruned_size:>15.2f}")
    print(f"{'Param Count':<25}{orig_params}{pruned_params:>15}")
    print(f"{'Avg Inference Time (s)':<25}{orig_time:.4f}{pruned_time:>15.4f}")

if __name__ == '__main__':
    original_model_path = '/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/eval-EXP-20250612-164244/DARTS.pth'
    pruned_model_path = '/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/DARTS_pruned.pth'
    dataset_path = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset'

    compare_models(original_model_path, pruned_model_path, dataset_path)