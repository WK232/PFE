import time
from datetime import datetime
import numpy as np
import torch
from timm.data import ImageDataset
from torch import nn
from torchvision.datasets import CIFAR100, CIFAR10
from torch.autograd import Variable
import torchvision.transforms as transforms
from contextlib import suppress

torch.backends.cudnn.benchmark = True

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


class TimePredictor:
    def __init__(self, steps, most_recent=30, drop_first=True):
        self.init_time = time.time()
        self.steps = steps
        self.most_recent = most_recent
        self.drop_first = drop_first  # drop iter 0

        self.time_list = []
        self.temp_time = self.init_time

    def update(self):
        time_interval = time.time() - self.temp_time
        self.time_list.append(time_interval)

        if self.drop_first and len(self.time_list) > 1:
            self.time_list = self.time_list[1:]
            self.drop_first = False

        self.time_list = self.time_list[-self.most_recent:]
        self.temp_time = time.time()

    def get_pred_text(self):
        single_step_time = np.mean(self.time_list)
        end_timestamp = self.init_time + single_step_time * self.steps
        return datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')


def _concat(xs):
    return torch.cat([x.reshape(-1) for x in xs])


class Architect(object):
    def __init__(self, distiller, args, amp_autocast=suppress):
        super(Architect, self).__init__()
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.distiller = distiller
        self.args = args
        self.amp_autocast = amp_autocast
        self.optimizer = torch.optim.Adam(self.distiller.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        with self.amp_autocast():
            _, losses_dict = self.distiller(input, target)
            loss = sum(losses_dict.values())

        theta = _concat(self.distiller.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.distiller.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, self.distiller.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(torch.tensor(eta, device='cuda') * (moment + dtheta)))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.distiller.student.model_train = False
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):

        with self.amp_autocast():
            _, losses_dict = self.distiller(input_valid, target_valid)
            loss = sum(losses_dict.values())
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        with self.amp_autocast():
            _, losses_dict = unrolled_model(input_valid, target_valid)
            unrolled_loss = sum(losses_dict.values())
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.student.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(torch.tensor(eta, device='cuda') * ig.data)

        for v, g in zip(self.distiller.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.distiller.new()
        model_dict = self.distiller.state_dict()
        params, offset = {}, 0
        for k, v in self.distiller.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].reshape(v.size())
            offset += v_length
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.distiller.student.parameters(), vector):
            p.data.add_(R * v)
        with self.amp_autocast():
            _, losses_dict = self.distiller(input, target)
            loss = sum(losses_dict.values())
        grads_p = torch.autograd.grad(loss, self.distiller.arch_parameters())

        for p, v in zip(self.distiller.student.parameters(), vector):
            p.data.sub_(2 * R * v)
        with self.amp_autocast():
            _, losses_dict = self.distiller(input, target)
            loss = sum(losses_dict.values())
        grads_n = torch.autograd.grad(loss, self.distiller.arch_parameters())

        for p, v in zip(self.distiller.student.parameters(), vector):
            p.data.add_(R * v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
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


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


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
        train_transform.transforms.append(Cutout(args.cutout_length,
                                                 args.cutout_prob))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


class ImageNetInstanceSample(ImageDataset):
    """ Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, name, class_map, load_bytes, is_sample=False, k=4096, **kwargs):
        super().__init__(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.parser)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.parser[i]
                label[i] = target

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR100InstanceSample(CIFAR100, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR100.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 100
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


class CIFAR10InstanceSample(CIFAR10, ImageNetInstanceSample):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, train, is_sample=False, k=4096, **kwargs):
        CIFAR10.__init__(self, root, train, **kwargs)
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 10
            num_samples = len(self.data)

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[self.targets[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target = CIFAR100.__getitem__(self, index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index
