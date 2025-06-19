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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from custom_dataloader import CoherencePhaseSegmentationDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  dataset = CoherencePhaseSegmentationDataset(root_dir=args.data)
  num_train = len(dataset)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = DataLoader(dataset, batch_size=args.batch_size,
                           sampler=torch.utils.data.SubsetRandomSampler(indices[:split]),
                           pin_memory=True, num_workers=2)

  valid_queue = DataLoader(dataset, batch_size=args.batch_size,
                           sampler=torch.utils.data.SubsetRandomSampler(indices[split:]),
                           pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    if args.epochs-epoch<=1:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, ((coh, pha), target) in enumerate(train_queue):
    model.train()
    n = target.size(0)
    input = torch.cat([coh, pha], dim=1).cuda()

    # FIX: Ensure target shape is (B, H, W)
    if target.ndim == 4 and target.shape[1] == 1:
      target = target.squeeze(1).long().cuda()
    elif target.ndim == 4 and target.shape[1] > 1:
    # Multi-channel, likely one-hot encoded: take argmax to get class indices
      target = target.argmax(dim=1).long().cuda()
    else:
      target = target.long().cuda()

    input_search, target_search = next(iter(valid_queue))
    input_search = torch.cat([input_search[0], input_search[1]], dim=1).cuda()

    # FIX: same for search target
    target_search = target_search.squeeze(1).long().cuda() if target_search.ndim == 4 else target_search.long().cuda()

    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    preds = logits.argmax(dim=1)
    acc = (preds == target).float().mean().item() * 100
    objs.update(loss.item(), n)
    top1.update(acc, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, ((coh, pha), target) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = torch.cat([coh, pha], dim=1).cuda()

      # FIX: Ensure correct shape
    # Inside train() and infer()
    if target.ndim == 4 and target.shape[1] == 1:
      target = target.squeeze(1).long().cuda()
    elif target.ndim == 4 and target.shape[1] > 1:
    # Multi-channel, likely one-hot encoded: take argmax to get class indices
      target = target.argmax(dim=1).long().cuda()
    else:
      target = target.long().cuda()

    logits = model(input)
    logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)
    loss = criterion(logits, target)

    preds = logits.argmax(dim=1)
    acc = (preds == target).float().mean().item() * 100

    n = target.size(0)
    objs.update(loss.item(), n)
    top1.update(acc, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

