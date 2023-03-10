
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inCh, outCh, stride, norm):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=stride),
            #nn.BatchNorm2d(outCh),
            norm2d(outCh, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outCh, out_channels=outCh,
                      kernel_size=3, padding=1, stride=1),
            norm2d(outCh, norm)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inCh != outCh:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inCh, out_channels=outCh,
                          kernel_size=1, stride=stride),
                norm2d(outCh, norm)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def norm2d(outCh, norm):
    if norm == 'group_norm':
        return nn.GroupNorm(2, outCh, affine=True)
    elif norm == 'batch_norm':
        return nn.BatchNorm2d(outCh)
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d(outCh, affine=True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


