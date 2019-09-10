import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from utils import *
import numpy as np

from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module

from non_local_block import NONLocalBlock3D

__all__ = ['WideResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()

        self.cfg = cfg

        NFR = cfg.num_features_relation

        NG = cfg.num_graph
        N = cfg.num_boxes
        T = cfg.num_frames

        NFG = cfg.num_features_gcn
        NFG_ONE = NFG

        self.fc_rn_theta_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFR) for i in range(NG)])

        self.fc_gcn_list = torch.nn.ModuleList(
            [nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])

        if cfg.dataset_name == 'volleyball':
            self.nl_gcn_list = torch.nn.ModuleList(
                [nn.LayerNorm([T*N, NFG_ONE]) for i in range(NG)])
        else:
            self.nl_gcn_list = torch.nn.ModuleList(
                [nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape
        NFR = self.cfg.num_features_relation
        NG = self.cfg.num_graph
        NFG_ONE = NFG

        OH, OW = self.cfg.out_size
        pos_threshold = self.cfg.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (
            graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (
            graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(
            B, N, 2)  # B*T, N, 2

        graph_boxes_distances = calc_pairwise_distance_3d(
            graph_boxes_positions, graph_boxes_positions)  # B, N, N

        position_mask = (graph_boxes_distances > (pos_threshold*OW))

        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](
                graph_boxes_features)  # B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](
                graph_boxes_features)  # B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(
                graph_boxes_features_theta, graph_boxes_features_phi.transpose(1, 2))  # B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(NFR)

            # B*N*N, 1
            similarity_relation_graph = similarity_relation_graph.reshape(
                -1, 1)

            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B, N, N)

            relation_graph[position_mask] = -float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](torch.matmul(
                relation_graph, graph_boxes_features))  # B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](
                one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)

            graph_boxes_features_list.append(one_graph_boxes_features)

        graph_boxes_features = torch.sum(torch.stack(
            graph_boxes_features_list), dim=0)  # B, N, NFG

        return graph_boxes_features, relation_graph

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

def nonlocalnet(input_layer, input_channel, mode='embedded_gaussian'):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = NONLocalBlock3D(in_channels=input_channel, mode=mode)
        out = net(input_layer)
    else:
        net = NONLocalBlock3D(in_channels=input_channel, mode=mode)
        out = net(input_layer)
    return out

class WideBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, non_local=False):
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
        self.non_local = non_local

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
        if self.non_local:
            out = nonlocalnet(out, out.size(1), mode='embedded_gaussian')
        return out

class WideResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 cfg,
                 k=1,
                 GCN=False,
                 shortcut_type='B'):
        self.inplanes = 64
        self.cfg = cfg
        self.GCN = GCN
        self.non_local = cfg.non_local
        num_classes = cfg.num_activities
        self.crop_size = cfg.crop_size
        self.dropout = cfg.dropout

        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1))
        self.layer2 = self._make_layer(
            block, 128 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256 * k, layers[2], shortcut_type, stride=2, non_local=self.non_local)
        self.layer4 = self._make_layer(
            block, 512 * k, layers[3], shortcut_type, stride=2, non_local=self.non_local)
        self.layer5 = self._make_layer(block, 512 * k, layers[1], shortcut_type, stride=2)

        self.roi_align = RoIAlign(self.crop_size[0], self.crop_size[1])

        self.global_avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avgpool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.cls = nn.Linear(512 * k * block.expansion, num_classes)

        self.gcn_list = torch.nn.ModuleList(
            [GCN_Module(self.cfg) for i in range(self.cfg.gcn_layers)])

        self.fc_emb_frame = nn.Sequential(nn.Linear(512 * k * block.expansion, self.cfg.num_feature_boxes),
                                        nn.BatchNorm1d(self.cfg.num_feature_boxes),
                                        nn.ReLU(inplace=True))

        self.fc_emb_1 = nn.Linear(self.crop_size[0]*self.crop_size[1]*self.cfg.emb_features, self.cfg.num_feature_boxes)
        self.nl_emb_1 = nn.LayerNorm([self.cfg.num_feature_boxes])
        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)
        self.fc_activities = nn.Linear(self.cfg.num_features_gcn, self.cfg.num_activities)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, non_local=False):
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
            if non_local and i % blocks-1 == 0:
                layers.append(block(self.inplanes, planes, non_local=True))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_data):
        x, bbox_in = input_data
        batch_size = x.size()[0]
        num_frames = x.size()[2]
        num_bbox = bbox_in.size()[2]
        max_score = 0.0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool_2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # (8, 1024, 4, 14, 14)
        if self.cfg.roi_align:
            # Use ROI align to get feature maps for each bbox.
            # Then avg bbox feature map over time and do classification for group activity (ie one person is doing the activity for the group).
            roi_align_features, boxes_in_flat = self.run_roi_align(self.roi_align, x, bbox_in) #(8, 1024, 120, 7, 7)
            roi_align_features = roi_align_features.reshape(batch_size, num_frames, num_bbox, -1, self.crop_size[0], self.crop_size[1]) #(8, 10, 12, 1024, 7, 7)
            # Find nonlocal features in each bbox feature map across time.
            roi_align_nl_features = self.run_nonlocal_network(roi_align_features) # (8, 1024, 12, 10, 7, 7)
            # Avg across time (num frames).
            # roi_align_features = roi_align_features.permute(0, 3, 2, 1, 4, 5)
            roi_align_features_avg_time = torch.squeeze(torch.mean(roi_align_nl_features, dim=3), dim=3) #(8, 1024, 12, 7, 7)
            roi_align_features_avg_time = self.global_avgpool_2d(roi_align_features_avg_time.reshape(batch_size, -1, self.crop_size[0], self.crop_size[1])) #(8, 1024, 12)
            roi_align_features_avg_time = roi_align_features_avg_time.reshape(batch_size, -1, num_bbox)
            for i in range(roi_align_features_avg_time.size()[-1]):
                bbox_roi_feature_map = roi_align_features_avg_time[:, :, i] # (8, 1024)
                activities_scores = self.cls(bbox_roi_feature_map)
                if torch.max(activities_scores) > max_score:
                    best_activities_score = activities_scores
                    max_score = torch.max(activities_scores)
            if self.GCN:
                # Start the GCN
                roi_align_features = roi_align_features.reshape(batch_size, num_frames, num_bbox, -1) # (8, 10, 12, 1024*14*14)
                roi_align_features = self.fc_emb_1(roi_align_features) # (8, 10, 12, 512)
                roi_align_features = self.nl_emb_1(roi_align_features)
                roi_align_features = F.relu(roi_align_features)
                graph_boxes_features = roi_align_features.reshape(batch_size, num_frames*num_bbox, self.cfg.num_features_gcn)
                #(8, 120, 512)
                for i in range(len(self.gcn_list)):
                    graph_boxes_features, relation_graph = self.gcn_list[i](
                        graph_boxes_features, boxes_in_flat)
                graph_boxes_features = graph_boxes_features.reshape(batch_size, num_frames, num_bbox, self.cfg.num_features_gcn)
                roi_align_features = roi_align_features.reshape(batch_size, num_frames, num_bbox, self.cfg.num_feature_boxes)
                # Fuse features from GCN, roialign and I3D
                x = self.layer5(x)
                x = torch.squeeze(torch.squeeze(torch.squeeze(self.global_avgpool_3d(x), dim=-1), dim=-1), dim=-1)
                x = self.fc_emb_frame(x)
                x = self.nl_emb_1(x)
                x = F.relu(x)
                boxes_states = graph_boxes_features + roi_align_features
                boxes_states = self.dropout_global(boxes_states)
                boxes_states_pooled, _ = torch.max(boxes_states, dim=2)
                boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, self.cfg.num_features_gcn)

                activities_scores = self.fc_activities(boxes_states_pooled_flat) 
                activities_scores = activities_scores.reshape(batch_size, num_frames, -1)
                # Avg across the frames.
                activities_scores = torch.mean(activities_scores, dim=1).reshape(batch_size, -1) + self.cls(x)
            return best_activities_score
        else:
            x = torch.squeeze(torch.squeeze(torch.squeeze(self.global_avgpool_3d(x), dim=-1), dim=-1), dim=-1)
            if self.dropout:
                x = self.dropout_global(x)
            activities_scores = self.cls(x)
        return activities_scores
    
    def run_nonlocal_network(self, input_data):
        """
        Run nonlocal networks on the RoIAlign features.
        input_data: (B, num_frames, num_bbox, C, H, W). Must change axes to (B, C, num_bbox, num_frames, H, W).
        """
        num_bbox = input_data.size()[2]
        input_data = input_data.permute(0, 3, 2, 1, 4, 5)
        output_nl_features = []
        for i in range(num_bbox):
            output_nl_features.append(nonlocalnet(input_data[:, :, i, :, :, :], input_data.size()[1], mode='embedded_gaussian'))
        return torch.stack(output_nl_features)

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
        return boxes_features, boxes_in_flat

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