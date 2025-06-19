import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from model import Network
from torch.utils.data import DataLoader
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDatasetclear', help='location of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/DARTS_pruned.pth')
args = parser.parse_args()

# ----------------------------
# Helper: Remove pruning artifacts
# ----------------------------
def remove_pruning(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if "weight_orig" in key:
            base_key = key.replace("weight_orig", "weight")
            mask_key = key.replace("weight_orig", "weight_mask")
            new_state_dict[base_key] = state_dict[key] * state_dict[mask_key]
        elif "weight_mask" in key:
            continue
        elif key not in new_state_dict:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

# ----------------------------
# Visualization Function
# ----------------------------
def visualize_segmentation(input_tensor, pred_mask, target_mask, class_colors, mean_iou, iou_per_class, save_path="segmentation_inference_result_pruning.png"):
    input_tensor = input_tensor.cpu()
    coherence = input_tensor[0].numpy()
    phase = input_tensor[1].numpy()

    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    target_color = np.zeros((*target_mask.shape, 3), dtype=np.uint8)

    for i, color in enumerate(class_colors):
        pred_color[pred_mask == i] = color
        target_color[target_mask == i] = color

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(coherence, cmap='gray')
    axs[0].set_title('Coherence')
    axs[1].imshow(phase, cmap='gray')
    axs[1].set_title('Phase')
    axs[2].imshow(pred_color)
    axs[2].set_title('Prediction')
    axs[3].imshow(target_color)
    axs[3].set_title('Ground Truth')

    for ax in axs:
        ax.axis('off')

    title_text = f"Mean IoU: {mean_iou:.4f} | " + " | ".join([f"Class {i}: {iou:.4f}" for i, iou in enumerate(iou_per_class)])
    plt.suptitle(title_text, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# IoU Metric Function
# ----------------------------
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

# ----------------------------
# Inference + Visualization + IoU
# ----------------------------
def infer_and_visualize(args, genotype, test_loader, NUM_CLASSES, device):
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)

    # Load and clean pruned model
    checkpoint = torch.load(args.model_path, map_location=device)
    clean_state_dict = remove_pruning(checkpoint)
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    class_colors = [(0, 0, 0), (255, 0, 0)]  # class 0, class 1

    with torch.no_grad():
        for batch_idx, ((coh, pha), target) in enumerate(test_loader):
            coh = coh.to(device)
            pha = pha.to(device)
            input = torch.cat([coh, pha], dim=1)
            target = target.to(device)

            output = model(input)
            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)

            input_img = input[0].cpu()
            pred_np = preds[0].cpu().numpy()
            target_np = target[0].cpu().numpy()

            mean_iou, iou_per_class = compute_iou(pred_np, target_np, num_classes=NUM_CLASSES)
            visualize_segmentation(input_img, pred_np, target_np, class_colors, mean_iou, iou_per_class)

            print(f"Saved segmentation_inference_result_quantization.png for batch {batch_idx}")
            print(f"Mean IoU: {mean_iou:.4f}")
            for cls_id, iou in enumerate(iou_per_class):
                print(f" - Class {cls_id} IoU: {iou:.4f}")
            break  # Remove to evaluate all batches

# ----------------------------
# Run
# ----------------------------
if __name__ == '__main__':
    test_dataset = CoherencePhaseSegmentationDataset(
        '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset',
        transform=None,
        target_transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    genotype = getattr(genotypes, args.arch)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    infer_and_visualize(args, genotype, test_loader, NUM_CLASSES=2, device=device)
