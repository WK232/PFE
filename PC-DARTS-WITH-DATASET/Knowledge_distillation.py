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
from model import NetworkCIFAR as Network
from custom_dataloader import CoherencePhaseSegmentationDataset
from torch.utils.data import DataLoader, random_split

parser = argparse.ArgumentParser("knowledge_distillation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the data corpus')
parser.add_argument('--teacher_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/PC-DARTS-WITH-DATASET/eval-EXP-20250617-095107/PC-DARTS.pth')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--T', type=float, default=4.0)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--save', type=str, default='KD-EXP')
parser.add_argument('--arch', type=str, default='PCDARTS')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')

args = parser.parse_args()

args.save = 'kd-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(args.save, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def soft_cross_entropy(student_logits, teacher_logits, T):
    student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=1e4, neginf=-1e4)

    student_logits = torch.clamp(student_logits, -50, 50)
    teacher_logits = torch.clamp(teacher_logits, -50, 50)

    log_probs = F.log_softmax(student_logits / T, dim=1)
    probs = F.softmax(teacher_logits / T, dim=1).clamp(min=1e-6)
    return -(probs * log_probs).sum(dim=1).mean() * (T * T)

def train_distill(train_loader, student_model, teacher_model, optimizer, T, alpha):
    student_model.train()
    teacher_model.eval()

    criterion_ce = nn.CrossEntropyLoss()
    acc_meter = utils.AvgrageMeter()
    obj_meter = utils.AvgrageMeter()

    for step, (inputs, target) in enumerate(train_loader):
        coh, pha = inputs
        input = torch.cat([coh, pha], dim=1).cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            teacher_logits, _ = teacher_model(input)

        student_logits, _ = student_model(input)

        loss_soft = soft_cross_entropy(student_logits, teacher_logits, T)
        if student_logits.shape[2:] != target.shape[1:]:
            student_logits = F.interpolate(student_logits, size=target.shape[1:], mode='bilinear', align_corners=False)

        loss_hard = criterion_ce(student_logits, target)
        loss = alpha * loss_soft + (1 - alpha) * loss_hard

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
        optimizer.step()

        acc, _, _ = utils.pixel_metrics(student_logits, target, 2)
        n = input.size(0)
        acc_meter.update(acc, n)
        obj_meter.update(loss.item(), n)

        if step % 10 == 0:
            logging.info('Train step %03d acc %.4f loss %.4f', step, acc_meter.avg, obj_meter.avg)

    return acc_meter.avg, obj_meter.avg

def evaluate(model, valid_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    acc_meter = utils.AvgrageMeter()
    recall_meter = utils.AvgrageMeter()
    loss_meter = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, target) in enumerate(valid_loader):
            coh, pha = inputs
            input = torch.cat([coh, pha], dim=1).cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)

            if logits.shape[2:] != target.shape[1:]:
                logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(logits, target)

            acc, recall, _ = utils.pixel_metrics(logits, target, 2)
            n = input.size(0)
            acc_meter.update(acc, n)
            recall_meter.update(recall, n)
            loss_meter.update(loss.item(), n)

            if step % 10 == 0:
                logging.info('Eval step %03d acc %.4f recall %.4f loss %.4f', step, acc_meter.avg, recall_meter.avg, loss_meter.avg)

    return acc_meter.avg, recall_meter.avg, loss_meter.avg

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)

    genotype = eval("genotypes.%s" % args.arch)

    teacher_model = Network(36, 2, 20, False, genotype)
    teacher_model.drop_path_prob = args.drop_path_prob
    teacher_model.load_state_dict(torch.load(args.teacher_path))
    teacher_model = teacher_model.cuda()
    teacher_model.eval()

    student_model = Network(16, 2, 8, False, genotype)
    student_model.drop_path_prob = args.drop_path_prob
    student_model.apply(init_weights)
    student_model = student_model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(student_model))
    logging.info("param number = %d", utils.count_parameters(student_model))
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    full_dataset = CoherencePhaseSegmentationDataset(args.data)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_data, valid_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for epoch in range(args.epochs):
        logging.info('epoch %d', epoch)

        train_acc, train_loss = train_distill(train_loader, student_model, teacher_model, optimizer, args.T, args.alpha)
        logging.info('Train acc %.4f loss %.4f', train_acc, train_loss)

        val_acc, val_iou, val_loss = evaluate(student_model, valid_loader)
        logging.info('Val acc %.4f IoU %.4f loss %.4f', val_acc, val_iou, val_loss)

    torch.save(student_model.state_dict(), os.path.join(args.save, f'student_epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()
