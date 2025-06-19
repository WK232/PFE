import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from model import Network
import utils
import genotypes
from custom_dataloader import CoherencePhaseSegmentationDataset
import argparse
parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the dataset')
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
# --- Configuration ---
DATA_DIR = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset'
BATCH_SIZE = 4
GPU_ID = 0
ARCH = 'DARTS'
NUM_CLASSES = 2
QUANTIZED_MODEL_PATH = '/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/DARTS_8bit_quantized.pth'  # Path to your quantized model

# --- Logging setup ---
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Load and prepare model ---
def load_model():
    model = Network(args.init_channels, 2, args.layers, args.auxiliary, genotypes.DARTS)
    model.load_state_dict(torch.load(QUANTIZED_MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    return model

# --- Evaluation function (your existing one) ---
def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    acc_meter = utils.AvgrageMeter()
    recall_meter = utils.AvgrageMeter()
    f1_meter = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (inputs, target) in enumerate(valid_queue):
            coh, pha = inputs
            input = torch.cat([coh, pha], dim=1).cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            acc, recall, f1 = utils.pixel_metrics(logits, target, NUM_CLASSES)
            n = input.size(0)
            objs.update(loss.item(), n)
            acc_meter.update(acc, n)
            recall_meter.update(recall, n)
            f1_meter.update(f1, n)

            if step % 50 == 0:
                logging.info('valid %03d loss %e acc %f recall %f f1 %f', step, objs.avg, acc_meter.avg, recall_meter.avg, f1_meter.avg)

    return acc_meter.avg, objs.avg, recall_meter.avg, f1_meter.avg

# --- Main evaluation logic ---
def main():
    torch.cuda.set_device(GPU_ID)
    cudnn.benchmark = True
    cudnn.enabled = True

    logging.info('Using GPU: %d', GPU_ID)

    model = load_model()
    model = model.cuda()

    logging.info("Model size = %f MB", utils.count_parameters_in_MB(model))
    logging.info("Total parameters = %d", utils.count_parameters(model))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = CoherencePhaseSegmentationDataset(DATA_DIR)
    val_fraction = 0.2
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    _, valid_data = random_split(dataset, [train_size, val_size])

    valid_queue = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    acc, loss, recall, f1 = infer(valid_queue, model, criterion)
    logging.info("Quantized Model Evaluation:\nAccuracy: %.4f\nLoss: %.4f\nRecall: %.4f\nF1 Score: %.4f",
                 acc, loss, recall, f1)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    logging.info('Total evaluation time: %.2fs', end - start)
