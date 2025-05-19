import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .operations import *
from .genotypes import Genotype


class MixedOp(nn.Module):
    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleDict()
        self.len = len(PRIMITIVES)
        self.no_parameter_len = sum(1 for primitive in PRIMITIVES if
                                    primitive in ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none', 'noise'])   
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops[primitive] = op

    def forward(self, x, weights, model_train=True, p=0):
        if p > 0 and self.training and model_train and self.len != self.no_parameter_len and random.random() < p:
            i = random.randint(self.no_parameter_len, self.len - 1)
            weights = torch.zeros_like(weights)
            weights[i] = 1
        return sum(w * op(x) for w, op in zip(weights, self._ops.values()))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.primitives = primitives['primitives_reduct' if reduction else 'primitives_normal']
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        edge_index = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s0, s1, weights, model_train=True, p=0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j], model_train=model_train, p=p)
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, PRIMITIVES, dataset, steps=4,
                 multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
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
                'num_classes': 100, 'input_size': (3, 32, 32),
                'crop_pct': 1.0, 'interpolation': 'bilinear', 'crop_mode': 'center',
                'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970),
            }
        if num_classes is None:
            self.num_classes = self.pretrained_cfg['num_classes']
        else:
            self.num_classes = num_classes

        self.num_classes = num_classes
        self.PRIMITIVES = PRIMITIVES
        self.steps = steps
        self._C = C
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.channel = [C * stem_multiplier, C * multiplier, C * 2 * multiplier, C * 4 * multiplier]
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, PRIMITIVES)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        self._initialize_alphas()
        self.model_train = True
        self.p = 0.1

    def new(self):  # copy arch_parameters to new model
        model_new = Network(self._C, self.num_classes, self._layers, self.PRIMITIVES).cuda()
        for new_param, param in zip(model_new.arch_parameters(), self.arch_parameters()):
            new_param.data.copy_(param.data.clone())
            new_param.requires_grad = param.requires_grad
        return model_new

    def forward(self, image, requires_feat=False):
        features = []
        s0 = s1 = self.stem(image)
        features.append(s1)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights, model_train=self.model_train, p=self.p)
        features.append(s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        features.append(out)
        logits = self.classifier(out)
        return (logits, features) if requires_feat else logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def stage_info(self, stage):
        if stage == 1:
            index = 0
            shape = (self.channel[0], 32, 32)  # 64
        elif stage == 2:
            index = 1
            shape = (self.channel[1], 32, 32)  # 64
        elif stage == 3:
            index = 2
            shape = (self.channel[2], 16, 16)  # 128
        elif stage == 4:
            index = 3
            shape = (self.channel[3], 8, 8)  # 256
        elif stage == -1:
            index = -1
            shape = self.channel[3]
        else:
            raise RuntimeError(f'Stage {stage} out of range (1-4)')
        return index, shape

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct']
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError:  # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start + j][k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
