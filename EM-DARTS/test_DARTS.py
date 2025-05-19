import argparse
import logging
from contextlib import suppress
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
_logger = logging.getLogger('test')

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

# DARTS Model parameters
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')  # 16
parser.add_argument('--layers', type=int, default=20, help='total number of layers')  # 8
parser.add_argument('--num_classes', type=int, default=10, metavar='N', help='number of label classes')


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
    setup_default_logging(log_path='test.log')
    args, args_text = _parse_args()
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

    if args.model == 'NetworkImageNet':
        model = NetworkImageNet(args.init_channels, args.num_classes, args.layers, genotype, args.dataset,
                                args.drop_path_prob)
    else:
        model = NetworkCIFAR(args.init_channels, args.num_classes, args.layers, genotype, args.dataset,
                             args.drop_path_prob)
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

    if args.dataset == 'cifar10':
        _, valid_transform = _data_transforms_cifar10(args)
        valid_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                                  transform=valid_transform)
    elif args.dataset == 'cifar100':
        _, valid_transform = _data_transforms_cifar100(args)

        valid_data = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True,
                                                   transform=valid_transform)
    else:
        _, valid_transform = _data_transforms_svhn(args)
        valid_data = torchvision.datasets.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    valid_acc, valid_obj = infer(valid_queue, model, validate_loss_fn, args, amp_autocast=amp_autocast)
    _logger.info('valid_acc %f - valid_obj %f', valid_acc, valid_obj)


def infer(valid_queue, model, criterion, args, amp_autocast=suppress):
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
                    'Valid: [{:>4d}/{}]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                    .format(
                        step, last_idx,
                        top1=top1, top5=top5,
                        loss=objs
                    ))
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
