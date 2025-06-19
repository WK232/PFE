import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes
import utils


parser = argparse.ArgumentParser("test iou class 1 and save best 5")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--arch', type=str, default='PCDARTS', help='architecture name')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--weights', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/eval-EXP-20250617-095107/PC-DARTS.pth', help='path to model weights')
parser.add_argument('--output_dir', type=str, default='best_predictions', help='directory to save visualizations')
args = parser.parse_args()


def compute_iou_class1(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    class_id = 1

    pred_inds = pred == class_id
    target_inds = target == class_id

    intersection = (pred_inds & target_inds).sum().item()
    union = (pred_inds | target_inds).sum().item()

    if union == 0:
        return float('nan')
    else:
        return intersection / union


def visualize_prediction(coh, pha, pred, target, idx, output_dir, iou, avg_iou):
    os.makedirs(output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(coh.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title("Coherence")
    axs[1].imshow(pha.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title("Phase")
    axs[2].imshow(pred.squeeze().cpu().numpy(), cmap='jet')
    axs[2].set_title("Prediction")
    axs[3].imshow(target.squeeze().cpu().numpy(), cmap='jet')
    axs[3].set_title("Ground Truth")

    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"IoU (class 1): {iou:.4f} | Avg IoU (class 1): {avg_iou:.4f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{idx}_iou_{iou:.4f}.png'))
    plt.close()


def main():
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    genotype = eval(f"genotypes.{args.arch}")
    model = Network(args.init_channels, 2, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = args.drop_path_prob
    model.load_state_dict(torch.load(args.weights))
    model.to(device)
    model.eval()

    dataset = CoherencePhaseSegmentationDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Inferencing on {len(dataset)} samples...")

    results = []
    ious = []

    with torch.no_grad():
        for idx, (inputs, target) in enumerate(loader):
            coh, pha = inputs
            coh, pha, target = coh.to(device), pha.to(device), target.to(device)
            input_tensor = torch.cat([coh, pha], dim=1)

            logits, _ = model(input_tensor)
            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

            preds = torch.argmax(logits, dim=1, keepdim=True)

            iou = compute_iou_class1(preds.cpu(), target.cpu())
            if not np.isnan(iou):
                ious.append(iou)
                results.append((iou, idx, coh.cpu(), pha.cpu(), preds.cpu(), target.cpu()))

            print(f"[{idx+1}/{len(loader)}] IoU class 1: {iou if not np.isnan(iou) else 'NaN'}")

    if len(ious) == 0:
        print("No valid samples (non-empty ground truths) found.")
        return

    avg_iou = np.mean(ious)
    print(f"\n✅ Average IoU for class 1 (excluding empty targets): {avg_iou:.4f}")

    # Sort results and save top 5
    top_results = sorted(results, key=lambda x: x[0], reverse=True)[:5]

    print("\n=== Top 5 IoU Results (class 1) ===")
    for i, (iou, idx, coh, pha, pred, target) in enumerate(top_results):
        print(f"Rank {i+1}: Sample {idx}, IoU = {iou:.4f}")
        visualize_prediction(coh, pha, pred, target, idx, args.output_dir, iou, avg_iou)

    print(f"\nSaved top 5 visualizations with IoU and average to: {args.output_dir}")


if __name__ == '__main__':
    main()
