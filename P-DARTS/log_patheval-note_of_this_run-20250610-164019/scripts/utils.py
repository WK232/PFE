import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def recall(output, target, average='macro'):
    with torch.no_grad():
        _, preds = torch.max(output, 1)
        num_classes = output.size(1)
        recall_scores = []

        for class_idx in range(num_classes):
            true_positives = ((preds == class_idx) & (target == class_idx)).sum().float()
            actual_positives = (target == class_idx).sum().float()
            if actual_positives == 0:
                recall_class = torch.tensor(0.0).to(output.device)
            else:
                recall_class = true_positives / actual_positives
            recall_scores.append(recall_class)

        recall_scores = torch.stack(recall_scores)

        if average == 'macro':
            return recall_scores.mean() * 100
        else:
            return recall_scores * 100

def precision_recall_f1(output, target, average='macro'):
    with torch.no_grad():
        _, preds = torch.max(output, 1)
        num_classes = output.size(1)
        precisions, recalls, f1s = [], [], []

        for class_idx in range(num_classes):
            true_positives = ((preds == class_idx) & (target == class_idx)).sum().float()
            predicted_positives = (preds == class_idx).sum().float()
            actual_positives = (target == class_idx).sum().float()

            precision = true_positives / predicted_positives if predicted_positives > 0 else torch.tensor(0.0).to(output.device)
            recall = true_positives / actual_positives if actual_positives > 0 else torch.tensor(0.0).to(output.device)

            if precision + recall == 0:
                f1 = torch.tensor(0.0).to(output.device)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        precisions = torch.stack(precisions) * 100
        recalls = torch.stack(recalls) * 100
        f1s = torch.stack(f1s) * 100

        if average == 'macro':
            return precisions.mean(), recalls.mean(), f1s.mean()
        else:
            return precisions, recalls, f1s

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def pixel_metrics(preds, targets, num_classes, ignore_index=None):
    """
    Calculate pixel-wise accuracy, recall, and F1 score.

    Args:
        preds: tensor of shape [B, C, H, W] (raw logits or probabilities)
        targets: tensor of shape [B, H, W] with class indices (ground truth)
        num_classes: int, number of classes
        ignore_index: int or None, label to ignore in evaluation

    Returns:
        accuracy, recall, f1_score (all floats)
    """
    with torch.no_grad():
        # If preds are logits, convert to predicted class indices
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)  # [B, H, W]

        preds = preds.view(-1)
        targets = targets.view(-1)

        if ignore_index is not None:
            mask = targets != ignore_index
            preds = preds[mask]
            targets = targets[mask]

        correct = preds.eq(targets).sum().item()
        total = targets.numel()
        accuracy = correct / total if total > 0 else 0

        # For recall and F1, compute per-class true positives, false negatives, false positives
        recall_sum = 0.0
        f1_sum = 0.0
        classes_with_gt = 0

        for cls in range(num_classes):
            pred_inds = preds == cls
            target_inds = targets == cls

            tp = (pred_inds & target_inds).sum().item()
            fn = (~pred_inds & target_inds).sum().item()
            fp = (pred_inds & ~target_inds).sum().item()

            if (tp + fn) > 0:
                recall_cls = tp / (tp + fn)
                precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0

                recall_sum += recall_cls
                f1_sum += f1_cls
                classes_with_gt += 1

        recall = recall_sum / classes_with_gt if classes_with_gt > 0 else 0
        f1 = f1_sum / classes_with_gt if classes_with_gt > 0 else 0

        return accuracy, recall, f1