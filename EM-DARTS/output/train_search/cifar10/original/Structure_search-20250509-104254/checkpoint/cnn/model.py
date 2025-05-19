import torch
import torch.nn as nn
from .operations import *
from .utils import drop_path


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob=0.):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers,  genotype, dataset, drop_path_prob=0.):
        super(NetworkCIFAR, self).__init__()
        if dataset == 'cifar10':
            self.pretrained_cfg = {
                'num_classes': 10, 'input_size': (3, 32, 32),
                'crop_pct': 1.0, 'interpolation': 'bilinear', 'crop_mode': 'center',
                'mean': (0.49139968, 0.48215827, 0.44653124), 'std': (0.24703233, 0.24348505, 0.26158768),
            }
        elif dataset == 'cifar100':
            self.pretrained_cfg = {
                'num_classes': 100, 'input_size': (3, 32, 32),
                'crop_pct': 1.0, 'interpolation': 'bilinear', 'crop_mode': 'center',
                'mean': (0.5070, 0.4865, 0.4409), 'std': (0.2673, 0.2564, 0.2761),
            }
        else:
            self.pretrained_cfg = {
                'num_classes': 10, 'input_size': (3, 32, 32),
                'crop_pct': 1.0, 'interpolation': 'bilinear', 'crop_mode': 'center',
                'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970),
            }
        if num_classes is None:
            self.num_classes = self.pretrained_cfg['num_classes']
        else:
            self.num_classes = num_classes
        self._layers = layers
        self.drop_path_prob = drop_path_prob
        self.feature_layers = [layers // 3 - 1, 2 * layers // 3 - 1, layers - 1]
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        self.channel = [stem_multiplier * C, C*cell.multiplier, C*2*cell.multiplier, C*4*cell.multiplier]

    def forward(self, image, requires_feat=False, requires_auxiliary=False):
        logits_aux = None
        features = []
        s0 = s1 = self.stem(image)
        features.append(s1)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i in self.feature_layers:
                features.append(s1)
            if i == 2 * self._layers // 3:
                logits_aux = s1
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        features.append(out)
        logits = self.classifier(out)

        if requires_auxiliary and requires_feat:
            output = logits, logits_aux, features
        elif requires_auxiliary and not requires_feat:
            output = logits, logits_aux
        elif not requires_auxiliary and requires_feat:
            output = logits, features
        else:
            output = logits
        return output

    def stage_info(self, stage):
        if stage == 1:
            index = 0
            shape = (self.channel[0], 32, 32)
        elif stage == 2:
            index = 1
            shape = (self.channel[1], 32, 32)
        elif stage == 3:
            index = 2
            shape = (self.channel[2], 16, 16)
        elif stage == 4:
            index = 3
            shape = (self.channel[3], 8, 8)
        elif stage == -1:
            index = -1
            shape = self.channel[3]
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
        return index, shape


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers,  genotype, drop_path_prob=0.):
        super(NetworkImageNet, self).__init__()

        self.pretrained_cfg = {
            'num_classes': 1000, 'input_size': (3, 224, 224),
            'crop_pct': 0.875, 'interpolation': 'bilinear', 'crop_mode': 'center',
            'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        }

        if num_classes is None:
            self.num_classes = self.pretrained_cfg['num_classes']
        else:
            self.num_classes = num_classes
        self._layers = layers
        self.drop_path_prob = drop_path_prob
        self.feature_layers = [layers // 3 - 1, 2 * layers // 3 - 1, layers - 1]
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),   #112
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1, bias=False),  #56
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False), #28
            nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        self.channel = [C, C*cell.multiplier, C*2*cell.multiplier, C*4*cell.multiplier]

    def forward(self, image, requires_feat=False, requires_auxiliary=False):
        features = []
        s0 = self.stem0(image)  # 224x224  112x112 56x56
        s1 = self.stem1(s0)  # 28x28
        features.append(s1)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i in self.feature_layers:
                features.append(s1)
            if i == 2 * self._layers // 3:
                logits_aux = s1
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        features.append(out)
        logits = self.classifier(out)
        if requires_auxiliary and requires_feat:
            output = logits, logits_aux, features
        elif requires_auxiliary and not requires_feat:
            output = logits, logits_aux
        elif not requires_auxiliary and requires_feat:
            output = logits, features
        else:
            output = logits
        return output

    def stage_info(self, stage):
        if stage == 1:
            index = 0
            shape = (self.channel[0], 28, 28)
        elif stage == 2:
            index = 1
            shape = (self.channel[1], 28, 28)
        elif stage == 3:
            index = 2
            shape = (self.channel[2], 14, 14)
        elif stage == 4:
            index = 3
            shape = (self.channel[3], 7, 7)
        elif stage == -1:
            index = -1
            shape = self.channel[3]
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
        return index, shape