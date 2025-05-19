import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .cell_operations import ResNetBasicblock, OPS


class MixedOp(nn.Module):
    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleDict()
        self.len = len(PRIMITIVES)
        self.no_parameter_len = sum(
            1 for primitive in PRIMITIVES if primitive in ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'none'])
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, C, stride, affine=False, track_running_stats=False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops[primitive] = op

    def forward(self, x, weights, model_train=True, p=0):
        if self.training and model_train and self.len != self.no_parameter_len and random.random() < p:
            i = random.randint(self.no_parameter_len, self.len - 1)
            weights = torch.zeros_like(weights)
            weights[i] = 1
        out = sum(w * op(x) for w, op in zip(weights, self._ops.values()))
        return out


class Cell(nn.Module):
    def __init__(self, steps, C, primitives):
        super(Cell, self).__init__()
        self.primitives = primitives['primitives_normal']
        self._steps = steps  # 3
        self._ops = nn.ModuleList()
        self.preprocess0 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=False)
        )
        edge_index = 0
        for i in range(self._steps):
            for j in range(1 + i):
                op = MixedOp(C, 1, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1

    def forward(self, s1, weights, model_train=True, p=0):
        s1 = self.preprocess0(s1)
        states = [s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j], model_train=model_train, p=p)
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return states[-1]


class NAS_Network(nn.Module):
    def __init__(self, C, num_classes, layers, primitives, steps=3):
        super(NAS_Network, self).__init__()
        self.pretrained_cfg = {
            'num_classes': 10, 'input_size': (3, 32, 32),
            'crop_pct': 1.0, 'interpolation': 'bilinear', 'crop_mode': 'center',
            'mean': (0.49139968, 0.48215827, 0.44653124), 'std': (0.24703233, 0.24348505, 0.26158768),
        }
        if num_classes is None:
            self.num_classes = self.pretrained_cfg['num_classes']
        else:
            self.num_classes = num_classes
        self.PRIMITIVES = primitives
        self._C = C
        self._layers = layers
        self._steps = steps
        self._multiplier = steps
        self.reduction_layers = [5, 11]
        self.channel = [C, C, C * 2, C * 4]
        C_curr = C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C))
        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in self.reduction_layers:
                C_curr *= 2
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = Cell(steps, C_curr, primitives)
            self.cells += [cell]
            C_prev = C_curr
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        self._initialize_alphas()
        self.model_train = True
        self.p = 0.1

    def new(self):  # copy arch_parameters to new model
        model_new = NAS_Network(self._C, self.num_classes, self._layers, self.PRIMITIVES).cuda()
        for new_param, param in zip(model_new.arch_parameters(), self.arch_parameters()):
            new_param.data.copy_(param.data.clone())
            new_param.requires_grad = param.requires_grad
        return model_new

    def forward(self, image, requires_feat=False):
        features = []
        s1 = self.stem(image)
        weights = F.softmax(self.alphas_normal, dim=1)
        features.append(s1)
        for i, cell in enumerate(self.cells):
            if i in self.reduction_layers:
                features.append(s1)
                s1 = cell(s1)
            else:
                s1 = cell(s1, weights, model_train=self.model_train, p=self.p)
        features.append(s1)
        s1 = self.lastact(s1)
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        features.append(out)
        logits = self.classifier(out)
        return (logits, features) if requires_feat else logits

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        num_ops = len(self.PRIMITIVES['primitives_normal'][0])
        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
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
        alphas = self.alphas_normal
        edge_index = 0
        out = ''
        ops = self.PRIMITIVES['primitives_normal'][0]
        for i in range(self._steps):
            for j in range(1 + i):
                _, indices = alphas[edge_index].data.topk(k=2, largest=True)
                if ops[indices[0]] == 'none':
                    out += f'|{ops[indices[1]]}~{j}'
                else:
                    out += f'|{ops[indices[0]]}~{j}'
                edge_index += 1
            if i != self._steps - 1:
                out += '|+'
        out += '|'
        return out
