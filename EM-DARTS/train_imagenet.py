import argparse
import logging
import os
import time
from collections import OrderedDict, defaultdict
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import yaml
from timm.data import resolve_data_config
from timm.models import model_parameters
from timm.utils import *
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim.lr_scheduler import LambdaLR

from cnn import *
from distillers import get_distiller
from utils import TimePredictor

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# ------------------------------------- My params ---------------------------------------
# Basic parameters
parser.add_argument('--model', default='NetworkImageNet', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',  # last.pth.tar
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--arch', type=str, default='exp2_4', help='which architecture to use')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--num_classes', type=int, default=1000, metavar='N', help='number of label classes')
parser.add_argument('-b', '--batch_size', type=int, default=1024, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--use_amp', type=str, default='native', help='Using native Torch AMP')  # native apex
parser.add_argument('--distiller', default='Darts_Loss', type=str)
parser.add_argument('--gt-loss-weight', default=1., type=float)
parser.add_argument('--train_search', action='store_true', default=False,
                    help='Used to determine whether to perform a structure search')
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--clip_mode', type=str, default='norm')
# ---------------------------------------------------------------------------------------

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', default='./autodl-tmp',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='imagenet',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
# Model parameters
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=1.0, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
# Misc
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint-hist', type=int, default=5, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


# 创建闭包以传递 epochs 参数
def create_lr_lambda(epochs, learning_rate):
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5.0
        elif epochs - epoch > 5:
            return (epochs - 5 - epoch) / (epochs - 5)
        else:
            return (epochs - epoch) / ((epochs - 5) * 5)

    return lr_lambda


def main():
    setup_default_logging(log_path='train.log')
    args, args_text = _parse_args()
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
        _logger.info(
            'Training in distributed mode with multiple processes, 1 GPU per process. '
            f'Process {args.rank}, total {args.world_size}.'
        )
    else:
        args.device = torch.device('cuda:0')
        args.world_size = 1
        args.rank = 0  # global rank
        _logger.info('Training with a single process on 1 GPU.')

    Distiller = get_distiller(args.distiller)
    genotype = eval("genotypes.%s" % args.arch)
    if args.rank == 0:
        _logger.info(f'genotype = {genotype}')
    model = NetworkImageNet(args.init_channels, args.num_classes, args.layers, genotype)

    if args.initial_checkpoint:
        check = torch.load(args.initial_checkpoint, map_location='cpu')
        model.load_state_dict(check['state_dict'])
    teacher = None
    resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # create the train and eval datasets
    data_dir = os.path.join(args.data_dir, 'imagenet')
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_eval = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_eval,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False
    )

    # Data loaders
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=6,
    )

    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        sampler=valid_sampler,
        pin_memory=True,
        prefetch_factor=6,
    )
    # setup loss function
    train_loss_fn = CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    validate_loss_fn = nn.CrossEntropyLoss().to(args.device)

    distiller = Distiller(model, teacher=teacher, criterion=train_loss_fn, args=args).to(args.device)
    if args.rank == 0:
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

    lr_lambda = create_lr_lambda(args.epochs, args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    if scheduler is not None and start_epoch > 0:
        scheduler.step(start_epoch)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    if args.use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if args.rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # setup distributed training
    if args.distributed:
        if args.rank == 0:
            _logger.info("Using native Torch DistributedDataParallel.")
        distiller = NativeDDP(distiller, device_ids=[args.local_rank], output_device=args.local_rank,
                              broadcast_buffers=True)

    if args.rank == 0:
        _logger.info('Scheduled epochs: {}'.format(args.epochs))

    # setup checkpoint saver and eval metric tracking
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        exp_name = '-'.join(['Structure_train',
                             datetime.now().strftime("%Y%m%d-%H%M%S"),
                             ])
        output_dir = get_outdir(f'./output/train/imagenet', exp_name)
        saver_dir = os.path.join(output_dir, 'checkpoint')
        os.makedirs(saver_dir, exist_ok=True)
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, checkpoint_dir=saver_dir,
            recovery_dir=saver_dir, decreasing=False, amp_scaler=loss_scaler,
            max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    try:
        tp = TimePredictor(args.epochs - start_epoch)
        for epoch in range(start_epoch, args.epochs):
            lr = scheduler.get_last_lr()[0]
            if args.drop_path_prob > 0:
                distiller.module.student.drop_path_prob = args.drop_path_prob * epoch / args.epochs
            train_metrics = train_one_epoch(
                epoch, distiller, loader_train, optimizer, args,
                amp_autocast=amp_autocast, lr=lr, loss_scaler=loss_scaler)

            scheduler.step()
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(distiller, args.world_size, args.dist_bn == 'reduce')
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics['top1']
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            tp.update()
            if args.rank == 0:
                print(f'Will finish at {tp.get_pred_text()}')
                print(f'Avg running time of latest {len(tp.time_list)} epochs: {np.mean(tp.time_list):.2f}s/ep.')

            if args.distributed:
                dist.barrier()
    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if args.rank == 0:
        os.system(f'mv train.log {output_dir}')


def train_one_epoch(
        epoch, distiller, loader, optimizer, args,
        amp_autocast=suppress, lr=None,
        loss_scaler=None):
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_gt_m = AverageMeter()
    losses_kd_m = AverageMeter()
    losses_m_dict = defaultdict(AverageMeter)
    if args.distributed:
        distiller.module.train()
    else:
        distiller.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        with amp_autocast():
            output, losses_dict = distiller(input, target)
            loss = sum(losses_dict.values())
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            for k in losses_dict:
                losses_m_dict[k].update(losses_dict[k].item(), input.size(0))
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_loss_dict = {}
                for k in losses_dict:
                    reduced_loss_dict[k] = reduce_tensor(losses_dict[k].data, args.world_size)

                losses_m.update(reduced_loss.item(), input.size(0))
                for k in reduced_loss_dict:
                    losses_m_dict[k].update(reduced_loss_dict[k].item(), input.size(0))

            if args.rank == 0:
                losses_infos = []
                for k, v in losses_m_dict.items():
                    info = f'{k.capitalize()}: {v.val:#.4g} ({v.avg:#.3g})'
                    losses_infos.append(info)
                losses_info = '  '.join(losses_infos)

                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    '{losses_info} '
                    'LR: {lr:.3e}'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        loss_gt=losses_gt_m,
                        loss_kd=losses_kd_m,
                        losses_info=losses_info,
                        lr=lr))

        end = time.time()
    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
