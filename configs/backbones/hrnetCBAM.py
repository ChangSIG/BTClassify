import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.backbones.resnet import BasicBlock, Bottleneck


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HRModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_channels):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.blocks = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()

        for i in range(num_branches):
            layers = []
            for j in range(num_blocks[i]):
                layers.append(block(num_channels[i], num_channels[i]))
            self.blocks.append(nn.Sequential(*layers))
            self.cbam_blocks.append(CBAM(num_channels[i]))

    def forward(self, x):
        out = []
        for i in range(self.num_branches):
            out.append(self.blocks[i](x[i]))
            out[-1] = self.cbam_blocks[i](out[-1])
        return out


class HRNet_CBAM(nn.Module):
    def __init__(self, arch):
        super(HRNet_CBAM, self).__init__()
        self.arch = arch
        self.hr_modules = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        for stage_cfg in arch:
            num_modules, num_branches, block_name, num_blocks, num_channels = stage_cfg
            if block_name == 'BOTTLENECK':
                block = Bottleneck
            elif block_name == 'BASIC':
                block = BasicBlock
            else:
                raise ValueError(f"Invalid block name: {block_name}")
            module = HRModule(num_branches, block, num_blocks, num_channels)
            self.hr_modules.append(module)

    def forward(self, x):
        x = self.stem(x)
        x = [x]  # Wrap the tensor in a list for processing in HRModule
        for module in self.hr_modules:
            x = module(x)
        return x



def build_hrnet_cbam(arch_name):
    arch_zoo = {
        # num_modules, num_branches, block, num_blocks, num_channels
        'w18': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (18, 36)],
                [4, 3, 'BASIC', (4, 4, 4), (18, 36, 72)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (18, 36, 72, 144)]],
        'w30': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (30, 60)],
                [4, 3, 'BASIC', (4, 4, 4), (30, 60, 120)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (30, 60, 120, 240)]],
        'w32': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (32, 64)],
                [4, 3, 'BASIC', (4, 4, 4), (32, 64, 128)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (32, 64, 128, 256)]],
        'w40': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (40, 80)],
                [4, 3, 'BASIC', (4, 4, 4), (40, 80, 160)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (40, 80, 160, 320)]],
        'w44': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (44, 88)],
                [4, 3, 'BASIC', (4, 4, 4), (44, 88, 176)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (44, 88, 176, 352)]],
        'w48': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (48, 96)],
                [4, 3, 'BASIC', (4, 4, 4), (48, 96, 192)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (48, 96, 192, 384)]],
        'w64': [[1, 1, 'BOTTLENECK', (4,), (64,)],
                [1, 2, 'BASIC', (4, 4), (64, 128)],
                [4, 3, 'BASIC', (4, 4, 4), (64, 128, 256)],
                [3, 4, 'BASIC', (4, 4, 4, 4), (64, 128, 256, 512)]],
    }
    arch = arch_zoo[arch_name]
    model = HRNet_CBAM(arch)
    return model
