# -*- coding: utf-8 -*-
# from unicodedata import name
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath
import matplotlib.pyplot as plt
# from .gcn_lib import Grapher, act_layer
from configs.backbones.gcn_lib import Grapher, act_layer


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.1):
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


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
                                               nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
                                               )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return scaled


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
        self.model_init()
        self.out_shape = [channels[-3],
                          channels[-2],
                          channels[-1]]
        # print(self.out_shape)
        self.init_attn = Spatial_Attention(kernel_size=3)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, images_mask=None):
        init_attn_input = torch.ones_like(inputs)
        if images_mask is not None:
            init_attn_input = images_mask
        inputs_vis = inputs.clone()
        x = init_attn_input * inputs_vis + inputs_vis
        # x = self.stem(inputs) + self.pos_embed
        x = self.stem(x) + self.pos_embed
        c2 = None
        c3 = None
        c4 = None
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i == sum(self.blocks[:1]) - 1:
                c2 = x
            if i == sum(self.blocks[:2]):
                c3 = x
            if i == sum(self.blocks[:3]) + 1:
                c4 = x

        if self.training:
            init_attn_input = self.init_attn(inputs_vis)

        return c2, c3, c4, x

    # def forward(self, inputs, images_mask=None):
    #     inputs_vis = inputs.clone()
    #     if images_mask is not None:
    #         x = images_mask * inputs_vis + inputs_vis
    #     # x = self.stem(inputs) + self.pos_embed
    #     x = self.stem(x) + self.pos_embed
    #     c2 = None
    #     c3 = None
    #     c4 = None
    #     for i in range(len(self.backbone)):
    #         x = self.backbone[i](x)
    #         if i == sum(self.blocks[:1]) - 1:
    #             c2 = x
    #         if i == sum(self.blocks[:2]):
    #             c3 = x
    #         if i == sum(self.blocks[:3]) + 1:
    #             c4 = x
    #
    #     if self.training:
    #         images_mask = self.init_attn(inputs_vis)
    #     return c2, c3, c4, x


def visualize_feature_map(feature_map):
    # 获取特征图的尺寸和通道数
    batch_size, channels, height, width = feature_map.shape

    # 创建一个大图，用于显示每个通道的特征图
    fig, axs = plt.subplots(channels, batch_size, figsize=(10, 10))

    for i in range(channels):
        for j in range(batch_size):
            # 获取特定通道的特征图
            channel_map = feature_map[j, i, :, :]

            # 绘制特征图
            axs[i][j].imshow(channel_map, cmap='gray')
            axs[i][j].axis('off')

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0, hspace=0)

    # 显示图像
    # plt.show()
    plt.savefig('logs/GNNAttnE/feature_map.png')
    plt.close()


def tiny_gnn(pretrained=False, **kwargs):
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,6,2] # number of basic blocks in the backbone
    # channels = [48, 96, 240, 384] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings

    model = DeepGCN(**kwargs)
    return model


def small_gnn(pretrained=False, **kwargs):
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,6,2] # number of basic blocks in the backbone
    # channels = [80, 160, 400, 640] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings

    model = DeepGCN(**kwargs)
    return model


def medium_gnn(pretrained=False, **kwargs):
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,16,2] # number of basic blocks in the backbone
    # channels = [96, 182, 384, 768] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings

    model = DeepGCN(**kwargs)
    return model


def big_gnn(pretrained=False, **kwargs):
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,18,2] # number of basic blocks in the backbone
    # channels = [128, 256, 512, 1024] # number of channels of deep features

    model = DeepGCN(**kwargs)
    return model


def gnn(pretrained=False, **kwargs):
    version = kwargs.pop('version')
    if version == 'tiny':
        return tiny_gnn(pretrained, **kwargs)
    if version == 'small':
        return small_gnn(pretrained, **kwargs)
    if version == 'medium':
        return medium_gnn(pretrained, **kwargs)
    if version == 'big':
        return big_gnn(pretrained, **kwargs)

