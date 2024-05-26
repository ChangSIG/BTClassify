# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
import torch.nn.functional as F
from timm.models.layers import DropPath
# from .gcn_lib import Grapher, act_layer
from configs.backbones.gcn_lib import Grapher, act_layer, DyGraphConv2d as GCNConv
from configs.backbones.hrnet_cs import CBAM


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        # self.att_weight = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        # self.att_bias = torch.nn.Parameter(torch.Tensor(1))
        # self.reset_parameters()
        self.cbam = CBAM(in_channels)

    def reset_parameters(self):
        # self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_weight)
        torch.nn.init.zeros_(self.att_bias)

    def forward(self, x):
        x = self.conv(x)
        # B, C, H, W = x.shape
        # att_score = (x.reshape(B, C, -1) * self.att_weight).sum(-1) + self.att_bias
        # att_weight = torch.softmax(att_score, dim=-1)
        # out = (att_weight * x).sum(dim=1)
        out = self.cbam(x)
        out += x

        return F.relu(out)


class DeepGCN(torch.nn.Module):
    def __init__(self, num_k=9, conv='mr', bias=True, epsilon=0.2, stochastic=True, act='gelu', norm='batch',
                 emb_dims=1024, drop_path=0.0, blocks=[2, 2, 6, 2], channels=[64, 128, 256, 512], img_size=[224, 224]):
        super(DeepGCN, self).__init__()
        num_k = num_k
        act = act
        norm = norm
        bias = bias
        epsilon = epsilon
        stochastic = stochastic
        conv = conv
        emb_dims = emb_dims
        drop_path = drop_path
        self.blocks = blocks
        self.n_blocks = sum(self.blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(num_k, num_k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        h, w = img_size[0], img_size[1]
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], h // 4, w // 4))

        HW = h // 4 * w // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.att_blocks = nn.ModuleList([])
        for i in channels:
            self.att_blocks.append(AttentionBlock(i, i))

        self.model_init()
        self.out_shape = [channels[-3],
                          channels[-2],
                          channels[-1]]
        # print(self.out_shape)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        c2 = None
        c3 = None
        c4 = None
        j = 0
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i == sum(self.blocks[:1]) - 1:
                c2 = self.att_blocks[j](x)
                j += 1
            if i == sum(self.blocks[:2]):
                c3 = self.att_blocks[j](x)
                j += 1
            if i == sum(self.blocks[:3]) + 1:
                c4 = self.att_blocks[j](x)
                j += 1

        x = self.att_blocks[j](x)
        return c2, c3, c4, x


# class DeepGCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
#         super().__init__()
#         self.num_layers = num_layers
#
#         self.stem = Stem(in_channels, hidden_channels, hidden_channels)
#         self.att_blocks = ModuleList()
#         for i in range(num_layers - 2):
#             self.att_blocks.append(AttentionBlock(hidden_channels, hidden_channels))
#         self.downsample = Downsample(hidden_channels, hidden_channels, hidden_channels) if num_layers > 2 else None
#         self.layers = ModuleList()
#         self.layers.append(FFN(hidden_channels, hidden_channels, hidden_channels, 2))
#         for i in range(num_layers - 2):
#             self.layers.append(FFN(hidden_channels, hidden_channels, hidden_channels, 3))
#         self.layers.append(torch.nn.Linear(hidden_channels, out_channels))
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         x = self.stem(x, edge_index)
#         for i in range(self.num_layers - 2):
#             x = self.att_blocks[i](x, edge_index)
#         if self.downsample is not None:
#             x = self.downsample(x, edge_index)
#         for i in range(self.num_layers - 2):
#             x = self.layers[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.layers[-1](x)
#         return x
#
#
def tiny_gnn(num_classes=10, dropout=0.3):
    return DeepGCN(in_channels=3, hidden_channels=32, out_channels=num_classes, num_layers=4, dropout=dropout)


def small_gnn(num_classes=10, dropout=0.4):
    return DeepGCN(in_channels=3, hidden_channels=64, out_channels=num_classes, num_layers=5, dropout=dropout)


def medium_gnn(num_classes=10, dropout=0.4):
    return DeepGCN(in_channels=3, hidden_channels=96, out_channels=num_classes, num_layers=7, dropout=dropout)


def big_gnn(num_classes=10, dropout=0.5):
    # return DeepGCN(in_channels=3, hidden_channels=128, out_channels=num_classes, num_layers=7, dropout=dropout)
    return DeepGCN(drop_path=dropout)


def gnn(version='medium', num_classes=2, dropout=0.4):
    if version == 'tiny':
        return tiny_gnn(num_classes, dropout)
    elif version == 'small':
        return small_gnn(num_classes, dropout)
    elif version == 'medium':
        return medium_gnn(num_classes, dropout)
    elif version == 'big':
        return big_gnn(dropout=dropout)
    else:
        raise ValueError(f'Invalid version name: {version}')
