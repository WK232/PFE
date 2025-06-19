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
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset  # Replace with actual module

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 2

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  logging.info(f'genotype = {genotype}')
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotypes.PCDARTS)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  logging.info("param number = %d", utils.count_parameters(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  from torch.utils.data import random_split

  full_dataset = CoherencePhaseSegmentationDataset(args.data)
  logging.info('Dataset size: %d', len(full_dataset))
# Define the split ratio
  val_fraction = 0.2  # 20% for validation
  val_size = int(len(full_dataset) * val_fraction)
  train_size = len(full_dataset) - val_size

# Split
  train_data, valid_data = random_split(full_dataset, [train_size, val_size])

  train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
  valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj, train_recall, train_f1 = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj, valid_recall, valid_f1 = infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
        best_acc = valid_acc
    logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
  torch.save(model.state_dict(), os.path.join(args.save, "DARTS.pth"))
    

def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    acc_meter = utils.AvgrageMeter()
    recall_meter = utils.AvgrageMeter()
    f1_meter = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        coh, pha = input
        input = torch.cat([coh, pha], dim=1).cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        logits, logits_aux = model(input)
        # Ensure target is 3D
        if target.ndim == 2:
          target = target.unsqueeze(1)
        if target.ndim == 3:
          target_size = target.shape[1:]
        else:
          raise ValueError(f"Unexpected target shape: {target.shape}")

        if logits.shape[2:] != target_size:
          logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        loss = criterion(logits, target)

        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        acc, recall, f1 = utils.pixel_metrics(logits, target, num_classes=2)

        n = input.size(0)
        objs.update(loss.item(), n)
        acc_meter.update(acc, n)
        recall_meter.update(recall, n)
        f1_meter.update(f1, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e acc: %.4f recall: %.4f f1: %.4f', step, objs.avg, acc_meter.avg, recall_meter.avg, f1_meter.avg)

    return acc_meter.avg, objs.avg, recall_meter.avg, f1_meter.avg



def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  acc_meter = utils.AvgrageMeter()
  recall_meter = utils.AvgrageMeter()
  f1_meter = utils.AvgrageMeter()
  model.eval()

  for step, (inputs, target) in enumerate(valid_queue):
    coh, pha = inputs
    input = torch.cat([coh, pha], dim=1).cuda()
    target = target.cuda(non_blocking=True)

    outputs = model(input)
    logits, logits_aux = model(input)
    if logits.shape[2:] != target.shape[1:]:
        logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)
    loss = criterion(logits, target)


    if args.auxiliary:
      logits_aux = outputs[1]
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight * loss_aux

    acc, recall, f1 = utils.pixel_metrics(logits, target, 2)
    n = input.size(0)
    objs.update(loss.item(), n)
    acc_meter.update(acc, n)
    recall_meter.update(recall, n)
    f1_meter.update(f1, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e acc: %.4f recall: %.4f f1: %.4f', step, objs.avg, acc_meter.avg, recall_meter.avg, f1_meter.avg)

  return acc_meter.avg, objs.avg, recall_meter.avg, f1_meter.avg


if __name__ == '__main__':
  main() 

