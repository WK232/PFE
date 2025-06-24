import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from model import NetworkCIFAR as Network
from torch.utils.data import DataLoader
from custom_dataloader import CoherencePhaseSegmentationDataset
import genotypes
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import logging
import utils

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser("Evaluate Pruned Segmentation Model")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='Path to dataset')
parser.add_argument('--model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/model_pruned_0_6.pth', help='Path to pruned model')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--init_channels', type=int, default=36)
parser.add_argument('--layers', type=int, default=20)
parser.add_argument('--auxiliary', action='store_true', default=False)
parser.add_argument('--arch', type=str, default='PCDARTS')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning before evaluation')
parser.add_argument('--epochs', type=int, default=5, help='Number of fine-tuning epochs')
args = parser.parse_args()

# ----------------------------
# Metrics
# ----------------------------
def compute_iou_single_class(pred_mask, target_mask, class_id=1):
    pred_inds = (pred_mask == class_id)
    target_inds = (target_mask == class_id)
    intersection = np.logical_and(pred_inds, target_inds).sum()
    union = np.logical_or(pred_inds, target_inds).sum()
    if union == 0:
        return None
    return intersection / union

def calculate_accuracy(preds, target):
    correct = (preds == target).sum().item()
    total = target.numel()
    return correct / total

def visualize(input_tensor, pred_mask, target_mask, class_colors, iou_pos, model_name, acc, avg_iou, out_path):
    input_tensor = input_tensor.cpu()
    coherence = input_tensor[0].numpy()
    phase = input_tensor[1].numpy()
    pred_img = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    target_img = np.zeros((*target_mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(class_colors):
        pred_img[pred_mask == i] = color
        target_img[target_mask == i] = color
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    axs[0].imshow(coherence, cmap='gray')
    axs[0].set_title('Coherence')
    axs[1].imshow(phase, cmap='gray')
    axs[1].set_title('Phase')
    axs[2].imshow(pred_img)
    axs[2].set_title('Prediction')
    axs[3].imshow(target_img)
    axs[3].set_title('Ground Truth')
    for ax in axs:
        ax.axis('off')
    title = f"Model: {model_name}\nAcc: {acc:.4f} | Best IoU: {iou_pos:.4f} | Avg IoU: {avg_iou:.4f}"
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path)
    plt.close()

# ----------------------------
# Fine-tuning Function
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx, report_every=10):
    model.train()
    running_loss = 0
    step_loss = 0
    step_acc = 0
    step_count = 0

    loop = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_idx+1}")
    for step, ((coh, pha), target) in loop:
        coh, pha, target = coh.to(device), pha.to(device), target.to(device)
        input_tensor = torch.cat([coh, pha], dim=1)

        optimizer.zero_grad()
        logits = model(input_tensor)
        if isinstance(logits, tuple): logits = logits[0]
        logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, target.long())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = calculate_accuracy(preds, target)

        running_loss += loss.item()
        step_loss += loss.item()
        step_acc += acc
        step_count += 1

        if (step + 1) % report_every == 0:
            avg_step_loss = step_loss / step_count
            avg_step_acc = step_acc / step_count
            print(f"[Epoch {epoch_idx+1} | Step {step+1}] Avg Loss: {avg_step_loss:.4f} | Avg Acc: {avg_step_acc:.4f}")
            step_loss = 0
            step_acc = 0
            step_count = 0

    return running_loss / len(dataloader)

# ----------------------------
# Evaluation Function
# ----------------------------
def evaluate():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.to(device)
    print("param number = %d", utils.count_parameters(model))

    dataset = CoherencePhaseSegmentationDataset(args.data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.fine_tune:
        print("🔧 Starting fine-tuning...")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device, epoch_idx=epoch)
            print(f"✅ Epoch {epoch+1} finished. Avg Loss: {loss:.4f}")
        torch.save(model, "fine_tuned_model.pth")
        print("✅ Fine-tuned model saved to 'fine_tuned_model.pth'")

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    acc_meter = utils.AvgrageMeter()
    recall_meter = utils.AvgrageMeter()
    loss_meter = utils.AvgrageMeter()

    iou_sum = 0.0
    valid_iou_count = 0

    max_iou = -1
    best_input, best_pred, best_target = None, None, None
    class_colors = [(0, 0, 0), (255, 0, 0)]

    with torch.no_grad():
        for step, ((coh, pha), target) in enumerate(val_loader):
            coh, pha, target = coh.to(device), pha.to(device), target.to(device)
            input_tensor = torch.cat([coh, pha], dim=1)

            logits = model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]

            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(logits, target.long())

            acc, recall, _ = utils.pixel_metrics(logits, target, args.num_classes)
            n = input_tensor.size(0)
            acc_meter.update(acc, n)
            recall_meter.update(recall, n)
            loss_meter.update(loss.item(), n)

            preds = torch.argmax(logits, dim=1)

            for i in range(preds.size(0)):
                pred_np = preds[i].cpu().numpy()
                target_np = target[i].cpu().numpy()
                iou_pos = compute_iou_single_class(pred_np, target_np, class_id=1)

                if iou_pos is not None:
                    iou_sum += iou_pos
                    valid_iou_count += 1

                    if iou_pos > max_iou:
                        max_iou = iou_pos
                        best_input = input_tensor[i]
                        best_pred = pred_np
                        best_target = target_np

            if step % 10 == 0:
                logging.info('Eval step %03d acc %.4f recall %.4f loss %.4f', step, acc_meter.avg, recall_meter.avg, loss_meter.avg)

    avg_iou = iou_sum / valid_iou_count if valid_iou_count > 0 else 0.0

    if best_input is not None:
        visualize(
            best_input, best_pred, best_target,
            class_colors, max_iou,
            os.path.basename(args.model_path), acc_meter.avg, avg_iou,
            out_path="best_pruned_model_inference_PCmodel.png"
        )
        print(f"🖼️ Saved best image with Best IoU: {max_iou:.4f} | Avg IoU: {avg_iou:.4f}")
    else:
        print("⚠️ No valid samples with class 1 found.")

    print(f"\n=== Final Evaluation ===")
    print(f"Avg Accuracy: {acc_meter.avg:.4f}")
    print(f"Avg Recall: {recall_meter.avg:.4f}")
    print(f"Avg Loss: {loss_meter.avg:.4f}")
    print(f"Avg IoU (Class 1): {avg_iou:.4f}")



if __name__ == '__main__':
    evaluate()
