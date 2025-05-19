import argparse
import csv
import logging
import math
import os
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torchvision.datasets
import torchvision.utils
import yaml
from timm.data import resolve_data_config
from timm.utils import *
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
from cnn import *
from distillers import get_distiller
from utils import Architect, _data_transforms_cifar10, _data_transforms_cifar100, _data_transforms_svhn

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train_search')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='train_search')

parser.add_argument('--model', default='Network', type=str)
parser.add_argument('--teacher', default='', type=str)
parser.add_argument('--input_size', default=None, nargs=3, type=int, metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop_pct', default=None, type=float, metavar='N',
                    help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('data_dir', metavar='DIR', default='data', help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--p_max', default=0.125, type=float, metavar='N',
                    help='The probability of a horizon mutating into one parametric operation')

# KD parameters
parser.add_argument('--distiller', default='CrossEntropy', type=str)  # CrossEntropy
parser.add_argument('--gt-loss-weight', default=1., type=float)

parser.add_argument('--train-search', action='store_true', default=False,
                    help='Used to determine whether to perform a structure search')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-3, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--use_amp', type=str, default='',
                    help='Using native Torch AMP. Training in mixed precision')  # 'native'

parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')  # 17
parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='original', help='searching space to choose from')
parser.add_argument('--p_not_grow', action='store_true', default=False, help='p does not grow linearly')
parser.add_argument('--grow_mode', type=str, default='linear',
                    help='modes with increasing probability p from 0 to p_{max}')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    args.train_search = True
    exp_name = '-'.join(['Structure_search',
                         datetime.now().strftime("%Y%m%d-%H%M%S"),
                         ])
    output_dir = get_outdir(f'./output/train_search/{args.dataset}/{args.search_space}', exp_name)
    saver_dir = os.path.join(output_dir, 'checkpoint')
    os.makedirs(saver_dir)
    log_dir = os.path.join(output_dir, 'train_search.log')
    setup_default_logging(log_path=log_dir)
    _logger.info('Training with a single process on 1 GPUs.')
    if args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        amp_autocast = suppress
    Distiller = get_distiller(args.distiller)
    spaces = spaces_dict[args.search_space]
    if args.model == 'NAS_Network':
        model = NAS_Network(args.init_channels, args.num_classes, args.layers, spaces)
    else:
        model = Network(args.init_channels, args.num_classes, args.layers, spaces, args.dataset)

    resolve_data_config(vars(args), model=model, verbose=True)
    teacher = None
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    distiller = Distiller(model, teacher=teacher, criterion=train_loss_fn, args=args)
    distiller = distiller.cuda()
    student_params, extra_params = distiller.get_learnable_parameters()
    _logger.info(f'\n-------------------------------'
                 f'\nLearnable parameters'
                 f'\nStudent: {student_params / 1e6:.2f}M'
                 f'\nExtra: {extra_params / 1e6:.2f}M'
                 f'\n-------------------------------')
    optimizer = torch.optim.SGD(
        distiller.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.dataset == 'cifar10':
        train_transform, _ = _data_transforms_cifar10(args)
        train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                  transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, _ = _data_transforms_cifar100(args)
        train_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True,
                                                   transform=train_transform)
    else:
        train_transform, _ = _data_transforms_svhn(args)
        train_data = torchvision.datasets.SVHN(root=args.data_dir, split='train', download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(args.train_portion * num_train)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, prefetch_factor=2, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, prefetch_factor=2, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate_min)
    architect = Architect(distiller, args, amp_autocast=amp_autocast)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    _logger.info('Scheduled epochs: {}'.format(args.epochs))
    _logger.info(f'p_max: {args.p_max}')
    if args.p_not_grow:
        _logger.info(f'grow_mode: not grow')
    else:
        _logger.info(f'grow_mode: {args.grow_mode}')
    _logger.info(f'search_space = {args.search_space}')
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]

        if args.p_not_grow:
            distiller.student.p = args.p_max
        elif args.grow_mode == 'cosine':
            distiller.student.p = 0.5 * args.p_max * (1 - math.cos(math.pi * epoch / 50)) if epoch < 50 else args.p_max
        elif args.grow_mode == 'exp':
            distiller.student.p = args.p_max * (1 - np.exp(-np.log(49) / 49 * epoch)) if epoch < 50 else args.p_max
        elif args.grow_mode == 'linear':
            distiller.student.p = args.p_max * epoch / 50 if epoch < 50 else args.p_max
        elif args.grow_mode == 'early':
            if epoch < args.epochs / 3:
                distiller.student.p = args.p_max
            else:
                distiller.student.p = 0
        elif args.grow_mode == 'middle':
            if args.epochs / 3 <= epoch < 2 * args.epochs / 3:
                distiller.student.p = args.p_max
            else:
                distiller.student.p = 0
        elif args.grow_mode == 'late':
            if epoch >= 2 * args.epochs / 3:
                distiller.student.p = args.p_max
            else:
                distiller.student.p = 0

        train_acc, train_obj = train(train_queue, valid_queue, architect, distiller, optimizer, lr, epoch, args,
                                     amp_autocast=amp_autocast)

        _logger.info('train_acc %f', train_acc)
        scheduler.step()
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, validate_loss_fn, epoch, args,
                                     amp_autocast=amp_autocast)
        _logger.info('valid_acc %f', valid_acc)
        _logger.info(f'epoch = {epoch}   \n genotype = {model.genotype()}')
        _logger.info(f'alphas_normal = \n {F.softmax(model.alphas_normal, dim=1)}')
        if args.model != 'NAS_Network':
            _logger.info(f' alphas_reduct = \n {F.softmax(model.alphas_reduce, dim=1)}')


def train(train_queue, valid_queue, architect, distiller, optimizer, lr, epoch, args, amp_autocast=suppress):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    distiller.train()
    last_idx = len(train_queue) - 1
    for step, ((input, target), (input_search, target_search)) in enumerate(zip(train_queue, valid_queue)):
        n = input.size(0)
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        input_search, target_search = input_search.cuda(non_blocking=True), target_search.cuda(non_blocking=True)
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        distiller.student.model_train = True
        optimizer.zero_grad()
        with amp_autocast():
            logits, losses_dict = distiller(input, target, epoch=epoch)
            loss = sum(losses_dict.values())
        loss.backward()
        nn.utils.clip_grad_norm_(distiller.parameters(), args.grad_clip)
        optimizer.step()
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(acc1.item(), n)
        top5.update(acc5.item(), n)
        if step % args.report_freq == 0 or step == last_idx:  #
            _logger.info(
                'Train: {} [{:>4d}/{}]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                'LR: {lr:.3e}'
                .format(
                    epoch,
                    step, last_idx,
                    loss=objs,
                    top1=top1, top5=top5,
                    lr=lr))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch, args, amp_autocast=suppress):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    last_idx = len(valid_queue) - 1
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
            with amp_autocast():
                logits = model(input)
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(acc1.item(), n)
            top5.update(acc5.item(), n)

            if step % args.report_freq == 0 or step == last_idx:
                _logger.info(
                    'Valid: {} [{:>4d}/{}]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                    .format(
                        epoch,
                        step, last_idx,
                        top1=top1, top5=top5,
                        loss=objs
                    ))
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
