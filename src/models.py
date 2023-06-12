'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F

def norm2d(outCh, norm):
    if norm == 'group_norm':
        return nn.GroupNorm(2, outCh, affine=True)
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(outCh)
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d(outCh, affine=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = norm2d(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = norm2d(planes, norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm2d(self.expansion*planes, norm)
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n1 = norm2d(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = norm2d(planes, norm)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.n3 = norm2d(self.expansion*planes, norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm2d(self.expansion*planes, norm)
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = F.relu(self.n2(self.conv2(out)))
        out = self.n3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        self.num_classes = args.num_classes
        self.width = args.width
        self.norm = args.norm
        self.num_blocks = num_blocks
        self.channels = [64, 128, 256, 512]
        self.channels = [i * self.width for i in self.channels]
        self.in_planes = self.channels[0]

        self.conv1 = nn.Conv2d(args.num_channels, self.channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.n1 = norm2d(self.channels[0], self.norm)
        self.layer1 = self._make_layer(block, self.channels[0], num_blocks[0], stride=1, norm=self.norm)
        self.layer2 = self._make_layer(block, self.channels[1], num_blocks[1], stride=2, norm=self.norm)
        self.layer3 = self._make_layer(block, self.channels[2], num_blocks[2], stride=2, norm=self.norm)
        self.layer4 = self._make_layer(block, self.channels[3], num_blocks[3], stride=2, norm=self.norm)
        self.linear = nn.Linear(self.channels[3]*block.expansion, self.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3, 4, 6, 3], args)

def ResNet50(args):
    return ResNet(Bottleneck, [3, 4, 6, 3], args)

def ResNet101(args):
    return ResNet(Bottleneck, [3, 4, 23, 3], args)

def ResNet152(args):
    return ResNet(Bottleneck, [3, 8, 36, 3], args)
