#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
from src.residualBlock import ResidualBlock, norm2d, Bottleneck

class ResNet(nn.Module):
    def __init__(self, args, block_nums):
        super(ResNet, self).__init__()
        self.width = args.width
        self.block_nums = block_nums
        self.channels = [32, 64, 128, 256]
        self.channels = [i * self.width for i in self.channels]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=args.num_channels, out_channels=self.channels[0], kernel_size=3, padding=1, stride=1),
            norm2d(self.channels[0], args.norm),
            nn.ReLU(inplace=True)
        )
        self.layer_1 = self.make_layer(ResidualBlock, self.channels[0], self.channels[0], stride=1, norm=args.norm, block_num=self.block_nums[0])
        self.layer_2 = self.make_layer(ResidualBlock, self.channels[0], self.channels[1], stride=2, norm=args.norm, block_num=self.block_nums[1])
        self.layer_3 = self.make_layer(ResidualBlock, self.channels[1], self.channels[2], stride=2, norm=args.norm, block_num=self.block_nums[2])
        self.layer_4 = self.make_layer(ResidualBlock, self.channels[2], self.channels[3], stride=2, norm=args.norm, block_num=self.block_nums[3])
        self.avgpool = nn.AvgPool2d((3, 3), stride=2)
        self.fc = nn.Linear(self.channels[3] * 1 * 1, args.num_classes)

    def forward(self, x):
        fc_width = 256 * self.width
        x = self.conv1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        fc_input = x.view(-1, fc_width * 1 * 1)
        x = self.fc(fc_input)
        return x

    def make_layer(self, block, inCh, outCh, stride, norm, block_num=2):
        layers = []
        layers.append(block(inCh, outCh, stride, norm))
        for i in range(block_num - 1):
            layers.append(block(outCh, outCh, 1, norm))
        return nn.Sequential(*layers)

def ResNet18(args):
    return ResNet(args, [2, 2, 2, 2])


def ResNet34(args):
    return ResNet(args, [3, 4, 6, 3])

def get_model(name):
    return  {'resnet18': ResNet18,
             'resnet34': ResNet34}[name]

'''# General ResNet class from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class RawResNet(nn.Module):
    def __init__(self, args, block, num_blocks):
        super(RawResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm2d(64, args.norm),
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=args.norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=args.norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=args.norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=args.norm)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out'''

'''def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])'''
