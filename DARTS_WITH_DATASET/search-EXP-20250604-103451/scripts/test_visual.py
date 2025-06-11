import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn
from model import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader
from custom_dataloader import CoherencePhaseSegmentationDataset  # Replace with actual module

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, required=True, help='root directory of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(input_tensor, pred_mask, target_mask, class_colors, save_path="segmentation_inference_result.png"):
    """
    input_tensor: torch.Tensor of shape (2, H, W)
    pred_mask: numpy array (H, W)
    target_mask: numpy array (H, W)
    class_colors: list of RGB tuples
    save_path: str, path to save the output image
    """
    input_tensor = input_tensor.cpu()
    coherence = input_tensor[0].numpy()
    phase = input_tensor[1].numpy()

    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    target_color = np.zeros((*target_mask.shape, 3), dtype=np.uint8)

    for i, color in enumerate(class_colors):
        pred_color[pred_mask == i] = color
        target_color[target_mask == i] = color

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
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

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def infer_and_visualize(args, genotype, test_loader, NUM_CLASSES, device):
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)
    model.load_state_dict(torch.load('eval-EXP-20250603-154048/DARTS.pth', map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (coh, pha), target in test_loader:
            input = torch.cat([coh, pha], dim=1).to(device)
            target = target.to(device)

            output = model(input)
            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)

    # Define colors for each class (example: adjust per your classes)
    class_colors = [
        (0, 0, 0),        # class 0 - black (background)
        (255, 0, 0),      # class 1 - red
        (0, 255, 0),      # class 2 - green
        (0, 0, 255),      # class 3 - blue
        # add more if needed
    ]

    with torch.no_grad():
        for batch_idx, ((coh, pha), target) in enumerate(test_loader):
            coh = coh.to(device)
            pha = pha.to(device)
            input = torch.cat([coh, pha], dim=1)  # Shape: (B, 2, H, W)
            target = target.to(device)

            output = model(input)
            logits = output[0] if isinstance(output, tuple) else output
            preds = torch.argmax(logits, dim=1)

        # Visualize first image in batch only
            visualize_segmentation(input[0], preds[0].cpu().numpy(), target[0].cpu().numpy(), class_colors)
            print(f"Saved segmentation_inference_result.png for batch {batch_idx}")
            break  # visualize only one batch


# Usage example:
test_dataset = CoherencePhaseSegmentationDataset('/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset/12days/EcrinPark', transform=None, target_transform=None)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
genotype=genotypes.DARTS  # Replace with actual genotype if needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
infer_and_visualize(args, genotype, test_loader, 2, device)