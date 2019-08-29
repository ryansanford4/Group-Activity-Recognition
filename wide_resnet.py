import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module

__all__ = ['WideResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class WideBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WideBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class WideResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 k=1,
                 crop_size=(7, 7),
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 32
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            32,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.layer1 = self._make_layer(block, 32 * k, layers[0], shortcut_type)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))
        self.layer2 = self._make_layer(
            block, 64 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 128 * k, layers[2], shortcut_type, stride=1)
        self.layer4 = self._make_layer(
            block, 256 * k, layers[3], shortcut_type, stride=1)
        self.layer5 = self._make_layer(block, 256 * k, layers[1], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        # self.fc = nn.Linear(512 * k * block.expansion, num_classes)
        self.crop_size = crop_size

        self.roi_align = RoIAlign(crop_size[0], crop_size[1])

        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cls = nn.Linear(256 * k * block.expansion * 2, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, stride, stride),
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x, bbox_in = input_data
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool_2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        roi_align_features = self.run_roi_align(self.roi_align, x, bbox_in)
        # Frame data = (B, 1024, 4, 28, 28)
        # RoIAlign = (B, 1024, 120, 14, 14)
        x = self.layer5(x)
        # Frame data = (B, 1024, 4, 14, 14)
        roi_align_features = self.layer5(roi_align_features)
        # RoIAlign = (B, 1024, 120, 7, 7)
        x = self.layer5(x)
        # Frame data = (B, 1024, 4, 7, 7)
        roi_align_features = self.global_avgpool(roi_align_features)
        x = self.global_avgpool(x)
        cls_layer = torch.cat((x, roi_align_features), dim=1)
        cls_layer = torch.squeeze(torch.squeeze(torch.squeeze(cls_layer, dim=-1), dim=-1), dim=-1)
        return self.cls(cls_layer)

    def run_roi_align(self, roi_align, features_multiscale, bbox_in):
        B, C, T, H, W = features_multiscale.size()
        input_size = bbox_in.size()
        boxes_in_flat = torch.reshape(bbox_in, (-1, 4))  #B*T*N, 4
        features_multiscale_resize = torch.reshape(features_multiscale, (B*T, C, H, W))

        boxes_idx = [i * torch.ones(input_size[2], dtype=torch.int)   for i in range(input_size[0]*input_size[1]) ]
        boxes_idx = torch.stack(boxes_idx).to(device=bbox_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (input_size[0]*input_size[1]*input_size[2],))  #B*T*N,
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
    
        # RoI Align
        boxes_features = self.roi_align(features_multiscale_resize,
                                            boxes_in_flat,
                                            boxes_idx_flat)
        boxes_features = torch.reshape(boxes_features, (input_size[0], C, -1, self.crop_size[0], self.crop_size[1]))
        return boxes_features

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = WideResNet(WideBottleneck, [3, 4, 6, 3], **kwargs)
    return model