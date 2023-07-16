import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torch.distributions import Bernoulli
import torch.nn.functional as F
import re
from collections import OrderedDict
import os
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters



def meta_conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class MAML_ConvolutionalNeuralNetwork(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size = 64, wh_size = 1):

        super(MAML_ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params = None):

        features = self.features(inputs, params = self.get_subdict(params,'features'))
        features_flatten = features.view((features.size(0), -1))
        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))
        return features, logits


class ANIL_ConvolutionalNeuralNetwork(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size = 1):

        super(ANIL_ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):

        features = self.features(inputs)
        features_flatten = features.view((features.size(0), -1))
        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))
        return features, logits

class BOIL_ConvolutionalNeuralNetwork(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size = 1):

        super(BOIL_ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):

        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features_flatten = features.view((features.size(0), -1))
        logits = self.classifier(features_flatten)
        return features, logits

class AuxModel(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size=1):
        super(AuxModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features_flatten = features.view((features.size(0), -1))
        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))
        return features, logits


class ReconstructionBoil_ConvolutionalNeuralNetwork(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size = 1):

        super(ReconstructionBoil_ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):

        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        flaten_features = features.view((features.size(0), -1))
        logits = self.classifier(flaten_features)
        return features, logits


class DropBlock(MetaModule):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        # self.gamma = gamma
        # self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample(
                (batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            # print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            # print (block_mask.size())
            # print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        # print ("mask", mask[0][0])
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            # block_idxs += left_padding
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)

        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class MetaBasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(MetaBasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        out = self.bn1(out, params=self.get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
        out = self.bn2(out, params=self.get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=self.get_subdict(params, 'conv3'))
        out = self.bn3(out, params=self.get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x, params = self.get_subdict(params, 'downsample'))
        out += residual
        out = self.relu3(out)

        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(MetaModule):
    def __init__(self,algorithm, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, out_features=5,
                 wh_size=None):
        self.inplanes = 3
        super(ResNet, self).__init__()
        self.algorithm = algorithm

        if algorithm == 'MAML' or algorithm =='BOIL':
            blocks = [MetaBasicBlock, MetaBasicBlock, MetaBasicBlock, MetaBasicBlock]
            isMeta = True
        elif algorithm == 'ANIL':
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
            isMeta = False

        self.layer1 = self._make_layer(blocks[0], 64, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, isMeta = isMeta)
        self.layer2 = self._make_layer(blocks[1], 128, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size,  isMeta = isMeta)
        self.layer3 = self._make_layer(blocks[2], 256, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, isMeta = isMeta)
        self.layer4 = self._make_layer(blocks[3], 512, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size, isMeta = isMeta)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        if algorithm == "MAML" or algorithm == 'ANIL':
            self.classifier = MetaLinear(512 * wh_size * wh_size, out_features)
        elif algorithm == "BOIL":
            self.classifier = nn.Linear(512 * wh_size * wh_size, out_features)

        for m in self.modules():
            if isinstance(m, MetaConv2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, MetaBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, isMeta = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if isMeta:
                downsample = MetaSequential(
                    MetaConv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=1, bias=False),
                    MetaBatchNorm2d(planes * block.expansion, track_running_stats=False),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, track_running_stats=False),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        if isMeta:
            return MetaSequential(*layers)
        else:
            return nn.Sequential(*layers)

    def forward(self, x, params=None):
        if self.algorithm == 'MAML' or self.algorithm == "BOIL":
            x = self.layer1(x, params=self.get_subdict( params, 'layer1'))
            x = self.layer2(x, params=self.get_subdict(params, 'layer2'))
            x = self.layer3(x, params=self.get_subdict(params, 'layer3'))
            x = self.layer4(x, params=self.get_subdict(params, 'layer4'))
        elif self.algorithm == 'ANIL':
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        features = x.view((x.size(0), -1))
        if self.algorithm == "MAML" or self.algorithm == 'ANIL':
            logits = self.classifier(self.dropout(features), params=self.get_subdict(params, 'classifier'))
        elif self.algorithm == "BOIL":
            logits = self.classifier(self.dropout(features))
        return None, logits

class EmbeddingModule(MetaModule):

    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size = 1):

        super(EmbeddingModule, self).__init__()

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )
        self.linear = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params = None):
        features = self.features(inputs, params = self.get_subdict(params,'features'))
        features_flatten = features.view((features.size(0), -1))
        logit = self.linear(features_flatten, params = self.get_subdict(params,'linear'))
        # logit = nn.ReLU()(logit)
        logit = nn.Sigmoid()(logit)
        return logit

class Body_ConvolutionalNetwork(MetaModule):
    def __init__(self, in_channels, hidden_size=64):
        super(Body_ConvolutionalNetwork, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        return features

class Classifier_ConvolutionNetwork(MetaModule):
    def __init__(self, out_channels, hidden_size=64, wh_size=1):
        super(Classifier_ConvolutionNetwork, self).__init__()
        self.classifier = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):
        features_flatten = inputs.view(inputs.size(0), -1)
        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))
        return logits


class Rotation_Classifier_ConvolutionNetwork(MetaModule):
    def __init__(self, out_channels=4, hidden_size=64, wh_size=1):
        super(Rotation_Classifier_ConvolutionNetwork, self).__init__()
        self.classifier = MetaLinear(hidden_size * wh_size * wh_size, out_channels)

    def forward(self, inputs, params=None):
        features_flatten = inputs.view(inputs.size(0), -1)
        logits = self.classifier(features_flatten, params=self.get_subdict(params, 'classifier'))
        return logits

class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

class RelationNetWork(nn.Module):
    def __init__(self, args = None):
        super(RelationNetWork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)
        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding
        self.args = args
        if args.dataset == 'miniimagenet':
            self.wh_size = 5
        elif args.dataset == 'cifar_fs' or args.dataset == 'FC100':
            self.wh_size = 2
        elif args.dataset == 'omniglot':
            self.wh_size = 1

    def forward(self, x):

        x = x.view(-1, 64, self.wh_size, self.wh_size)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out

