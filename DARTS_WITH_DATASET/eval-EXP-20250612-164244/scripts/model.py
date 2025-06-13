import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path

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
            self._concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            self._concat = genotype.normal_concat

        self._steps = len(op_names) // 2
        self.multiplier = len(self._concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states.append(s)
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadSegmentation(nn.Module):
    """Auxiliary head for segmentation. Input assumed to be smaller feature map."""
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadSegmentation, self).__init__()
        self.aux = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, num_classes, 1)
        )

    def forward(self, x):
        return self.aux(x)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(Network, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(2, C_curr, 3, padding=1, bias=False),
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

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            # Save for auxiliary head (at 2/3 layers depth)
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadSegmentation(C_to_auxiliary, num_classes)

        # Final conv to output segmentation mask logits per class
        self.classifier = nn.Sequential(
            nn.Conv2d(C_prev, C_prev // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_prev // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_prev // 2, num_classes, 1)
        )

    def forward(self, x):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        logits = self.classifier(s1)  # Output segmentation logits [N, num_classes, H', W']

        # Optional: upsample output to input resolution if reduced spatial size
        # (Depends on how much reduction you allow in the network)
        logits = nn.functional.interpolate(logits, size=x.size()[2:], mode='bilinear', align_corners=False)

        return logits, logits_aux
