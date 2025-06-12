import argparse
import logging
import os
from contextlib import suppress
from datetime import datetime
import csv
import torch
from torch import nn
import torchvision.datasets
import torchvision.utils
import yaml
from timm.data import resolve_data_config
from timm.utils import *
from cnn import *
from distillers import get_distiller
from utils import _data_transforms_cifar10, _data_transforms_cifar100, _data_transforms_svhn

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
_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='cifar_train')
parser.add_argument('data_dir', metavar='DIR', default='data', help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--model', default='NetworkCIFAR', type=str)
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
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
parser.add_argument('-b', '--batch_size', type=int, default=96, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 4)')

parser.add_argument('--distiller', default='Darts_Loss', type=str)
parser.add_argument('--gt-loss-weight', default=1., type=float)

parser.add_argument('--train_search', action='store_true', default=False,
                    help='Used to determine whether to perform a structure search')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser.add_argument('--use_amp', type=str, default='', help='Using native Torch AMP')  # native
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
# DARTS Model parameters
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')  # 16
parser.add_argument('--layers', type=int, default=20, help='total number of layers')  # 8
parser.add_argument('--num_classes', type=int, default=10, metavar='N', help='number of label classes')
parser.add_argument('--arch', type=str, default='genotype1', help='which architecture to use')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')


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
    exp_name = '-'.join(['Structure_train',
                         datetime.now().strftime("%Y%m%d-%H%M%S"),
                         ])
    output_dir = get_outdir(f'./output/train/{args.dataset}', exp_name)
    saver_dir = os.path.join(output_dir, 'checkpoint')
    os.makedirs(saver_dir)
    log_dir = os.path.join(output_dir, 'train.log')
    setup_default_logging(log_path=log_dir)

    _logger.info('Training with a single process on 1 GPUs.')
    args.cutout = True
    if args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        amp_autocast = suppress
    genotype = eval("genotypes.%s" % args.arch)
    _logger.info(f'genotype = {genotype}')
    Distiller = get_distiller(args.distiller)
    model = NetworkCIFAR(args.init_channels, args.num_classes, args.layers, genotype, args.dataset, args.drop_path_prob)
    if args.initial_checkpoint:
        model.load_state_dict(torch.load(args.initial_checkpoint))

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
        0.025,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = _data_transforms_cifar10(args)
        train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                                  transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                                  transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = _data_transforms_cifar100(args)
        train_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True,
                                                   transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                                   transform=valid_transform)
    else:
        train_transform, valid_transform = _data_transforms_svhn(args)
        train_data = torchvision.datasets.SVHN(root=args.data_dir, split='train', download=True, transform=train_transform)
        valid_data = torchvision.datasets.SVHN(root=args.data_dir, split='test', download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=True, prefetch_factor=2, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False, prefetch_factor=2, pin_memory=True)
    _logger.info('Scheduled epochs: {}'.format(args.epochs))
    best_val_acc = 0
    records = []
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        # train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        train_acc, train_obj, recall_score, f1_score = train(train_queue, distiller, optimizer, lr, epoch, args, amp_autocast=amp_autocast)
        scheduler.step()
        valid_acc, valid_obj, recall_valid, f1_valid = infer(valid_queue, model, validate_loss_fn, epoch, args, amp_autocast=amp_autocast)
        records.append([epoch, train_acc, train_obj, valid_acc, valid_obj])
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(saver_dir, 'best.pth'))
        _logger.info('valid_acc %f - best_valid_acc %f', valid_acc, best_val_acc)
    torch.save(model.state_dict(), os.path.join(saver_dir, 'last.pth'))
    # os.system(f'mv train.log {output_dir}')

    with open(os.path.join(output_dir, 'records.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss'])  # 写入 CSV 文件的标题行
        writer.writerows(records)  # 写入数据记录


from sklearn.metrics import recall_score, f1_score
import torch.nn.functional as F

def train(train_queue, distiller, optimizer, lr, epoch, args, amp_autocast=suppress):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

    last_idx = len(train_queue) - 1
    for step, (input, target) in enumerate(train_queue):
        distiller.train()
        n = input.size(0)
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
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

        # ---- Compute recall and F1-score ----
        preds = torch.argmax(logits, dim=1)
        recall = recall_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        f1 = f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        recall_meter.update(recall, n)
        f1_meter.update(f1, n)
        # -------------------------------------

        if step % args.report_freq == 0 or step == last_idx:
            _logger.info(
                'Train: {} [{:>4d}/{}]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                'Recall: {recall:.4f}  F1: {f1:.4f}  '
                'LR: {lr:.3e}'
                .format(
                    epoch,
                    step, last_idx,
                    loss=objs,
                    top1=top1, top5=top5,
                    recall=recall, f1=f1,
                    lr=lr))

    return top1.avg, objs.avg, recall_meter.avg, f1_meter.avg



from sklearn.metrics import recall_score, f1_score

def infer(valid_queue, model, criterion, epoch, args, amp_autocast=suppress):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    recall_meter = AverageMeter()
    f1_meter = AverageMeter()

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

            # ---- Compute recall and F1-score ----
            preds = torch.argmax(logits, dim=1)
            recall = recall_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
            f1 = f1_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
            recall_meter.update(recall, n)
            f1_meter.update(f1, n)
            # -------------------------------------

            if step % args.report_freq == 0 or step == last_idx:
                _logger.info(
                    'Valid: {} [{:>4d}/{}]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Recall: {recall:.4f}  F1: {f1:.4f}'
                    .format(
                        epoch,
                        step, last_idx,
                        loss=objs,
                        top1=top1, top5=top5,
                        recall=recall, f1=f1
                    ))

    return top1.avg, objs.avg, recall_meter.avg, f1_meter.avg



if __name__ == '__main__':
    main()
