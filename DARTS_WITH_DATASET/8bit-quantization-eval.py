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

# --- Argument parsing ---
parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--data', type=str, default='/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset', help='location of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--init_channels', type=int, default=36, help='initial channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--arch', type=str, default='DARTS', help='architecture genotype')
parser.add_argument('--quantized_model_path', type=str, default='/home/kharratw/Documents/tessssst/PFE/DARTS_WITH_DATASET/DARTS_8bit_quantized.pth', help='path to 8-bit quantized model .pth')
args = parser.parse_args()

# --- Logging setup ---
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

NUM_CLASSES = 2

# --- Dequantization helper ---
def dequantize_tensor(q_weight, scale, zero_point):
    return scale * (q_weight.float() - zero_point)

# --- Load and dequantize model ---
def load_dequantized_model(model, quantized_path):
    q_state = torch.load(quantized_path, map_location='cpu')
    new_state = {}

    for name, value in q_state.items():
        if isinstance(value, dict) and all(k in value for k in ['q_weight', 'scale', 'zero_point']):
            # Dequantize Conv2d weight
            deq_tensor = dequantize_tensor(value['q_weight'], value['scale'], value['zero_point'])
            new_state[name] = deq_tensor
        elif isinstance(value, torch.Tensor):
            # Handle buffers like running_mean, running_var
            new_state[name] = value
        else:
            logger.info(f"Skipping unknown entry: {name}")
    
    model.load_state_dict(new_state, strict=True)   
    return model

# --- Inference ---
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
                logger.info('valid %03d loss %e acc %f recall %f f1 %f',
                             step, objs.avg, acc_meter.avg, recall_meter.avg, f1_meter.avg)

    return acc_meter.avg, objs.avg, recall_meter.avg, f1_meter.avg

# --- Main ---
def main():
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info('Using GPU: %d', args.gpu)

    # Build model and load quantized weights
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, getattr(genotypes, args.arch))
    model = load_dequantized_model(model, args.quantized_model_path)
    model = model.cuda()

    logger.info("Model size = %f MB", utils.count_parameters_in_MB(model))
    logger.info("Total parameters = %d", utils.count_parameters(model))

    criterion = nn.CrossEntropyLoss().cuda()

    # Load dataset and create validation loader
    dataset = CoherencePhaseSegmentationDataset(args.data)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, valid_data = random_split(dataset, [train_size, val_size])

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # Run evaluation
    acc, loss, recall, f1 = infer(valid_queue, model, criterion)
    logger.info("Quantized Model Evaluation:\nAccuracy: %.4f\nLoss: %.4f\nRecall: %.4f\nF1 Score: %.4f",
                 acc, loss, recall, f1)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    logger.info('Total evaluation time: %.2fs', end - start)
