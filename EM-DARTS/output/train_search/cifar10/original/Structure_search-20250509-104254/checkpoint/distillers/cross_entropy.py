from ._base import BaseDistiller
from .registry import register_distiller
# import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 7x7"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),  # 2x2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),  # 1x1
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


@register_distiller
class Darts_Loss(BaseDistiller):
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(Darts_Loss, self).__init__(student, teacher, criterion, args)
        self.criterion = criterion
        _, C = self.student.stage_info(-1)
        if args.model == 'NetworkImageNet':
            self.auxiliary_head = AuxiliaryHeadImageNet(C, args.num_classes)
        else:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C, args.num_classes)

    def forward(self, image, label, *args, **kwargs):
        logits_student, logits_aux = self.student(image, requires_feat=False, requires_auxiliary=True)
        loss = self.args.gt_loss_weight * self.criterion(logits_student, label)
        logits_aux = self.auxiliary_head(logits_aux)
        loss_aux = self.args.auxiliary_weight * self.criterion(logits_aux, label)
        losses_dict = {
            "loss_logits": loss,
            "loss_aux": loss_aux
        }
        return logits_student, losses_dict


@register_distiller
class CrossEntropy(BaseDistiller):
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CrossEntropy, self).__init__(student, teacher, criterion, args)

    def new(self):  # copy arch_parameters to new model
        if self.args.train_search:
            student = self.student.new()
            model_new = CrossEntropy(student, self.teacher, self.criterion, self.args).cuda()
            return model_new

    def arch_parameters(self):
        if self.args.train_search:
            return self.student.arch_parameters()

    def forward(self, image, label, *args, **kwargs):
        logits_student = self.student(image)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        losses_dict = {
            "loss_gt": loss_gt,
        }
        return logits_student, losses_dict
