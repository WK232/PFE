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
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

CIFAR_CLASSES = 10
# This must match the training config
genotype = eval("genotypes.%s" % args.arch)
model = Network(args.init_channels, CIFAR_CLASSES, args.layers, auxiliary=True, genotype=genotype)
model.load_state_dict(torch.load("DARTS.pth"))
model = model.cuda()
model.eval()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        if hasattr(module, 'weight'):
            weight = module.weight
            num_zeros = torch.sum(weight == 0).item()
            num_elements = weight.nelement()
            sparsity = 100.0 * num_zeros / num_elements
            print(f"{name}: {sparsity:.2f}% sparsity")

total_zero = 0
total_weights = 0

for module in model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        if hasattr(module, 'weight'):
            total_zero += torch.sum(module.weight == 0).item()
            total_weights += module.weight.nelement()

model_sparsity = 100.0 * total_zero / total_weights
print(f"Overall Model Sparsity: {model_sparsity:.2f}%")
import torch.nn.utils.prune as prune

    # Apply pruning
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

torch.save(model.state_dict(), "DARTS_pruned.pth")

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        if hasattr(module, 'weight'):
            weight = module.weight
            num_zeros = torch.sum(weight == 0).item()
            num_elements = weight.nelement()
            sparsity = 100.0 * num_zeros / num_elements
            print(f"{name}: {sparsity:.2f}% sparsity")

total_zero = 0
total_weights = 0

for module in model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        if hasattr(module, 'weight'):
            total_zero += torch.sum(module.weight == 0).item()
            total_weights += module.weight.nelement()

model_sparsity = 100.0 * total_zero / total_weights
print(f"Overall Model Sparsity: {model_sparsity:.2f}%")

print(model)
