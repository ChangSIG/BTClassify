import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..basic.build_layer import build_conv_layer, build_norm_layer
from ..common import BaseModule, ModuleList, Sequential
from .resnet import BasicBlock, Bottleneck, ResLayer, get_expansion
# from configs.basic.build_layer import build_conv_layer, build_norm_layer
# from configs.common import BaseModule, ModuleList, Sequential
# from configs.backbones.resnet import BasicBlock, Bottleneck, ResLayer, get_expansion


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':  # 平均池化，池化窗口大小与输入图像大小相同
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':  # 最大池化，池化窗口大小与输入图像大小相同
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)  # 将平均池化和最大池化的输出分别输入到MLP中，得到的结果进行相加
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output  # 将输入F和通道注意力模块的输出Mc相乘，得到F'


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

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


class HRModule(BaseModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.

    Args:
        num_branches (int): The number of branches.
        block (``BaseModule``): Convolution block module.
        num_blocks (tuple): The number of blocks in each branch.
            The length must be equal to ``num_branches``.
        num_channels (tuple): The number of base channels in each branch.
            The length must be equal to ``num_branches``.
        multiscale_output (bool): Whether to output multi-level features
            produced by multiple branches. If False, only the first level
            feature will be output. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        conv_cfg (dict, optional): Dictionary to construct and config conv
            layer. Defaults to None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to ``dict(type='BN')``.
        block_init_cfg (dict, optional): The initialization configs of every
            blocks. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_branches,
                 block,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 block_init_cfg=None,
                 init_cfg=None):
        super(HRModule, self).__init__(init_cfg)
        self.block_init_cfg = block_init_cfg
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, block, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            out_channels = num_channels[i] * get_expansion(block)
            branches.append(
                ResLayer(
                    block=block,
                    num_blocks=num_blocks[i],
                    in_channels=self.in_channels[i],
                    out_channels=out_channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp,
                    init_cfg=self.block_init_cfg,
                ))

        return ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsample the feature maps of smaller scales.
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    # Keep the feature map with the same scale.
                    fuse_layer.append(None)
                else:
                    # Downsample the feature maps of larger scales.
                    conv_downsamples = []
                    for k in range(i - j):
                        # Use stacked convolution layers to downsample.
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(BaseModule):

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}
    arch_zoo = {
        # num_modules, num_branches, block, num_blocks, num_channels
        'w18': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (18, 36)],
                [4, 3, 'BASIC',      (4, 4, 4),    (18, 36, 72)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (18, 36, 72, 144)]],
        'w30': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (30, 60)],
                [4, 3, 'BASIC',      (4, 4, 4),    (30, 60, 120)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (30, 60, 120, 240)]],
        'w32': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (32, 64)],
                [4, 3, 'BASIC',      (4, 4, 4),    (32, 64, 128)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (32, 64, 128, 256)]],
        'w40': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (40, 80)],
                [4, 3, 'BASIC',      (4, 4, 4),    (40, 80, 160)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (40, 80, 160, 320)]],
        'w44': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (44, 88)],
                [4, 3, 'BASIC',      (4, 4, 4),    (44, 88, 176)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (44, 88, 176, 352)]],
        'w48': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (48, 96)],
                [4, 3, 'BASIC',      (4, 4, 4),    (48, 96, 192)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (48, 96, 192, 384)]],
        'w64': [[1, 1, 'BOTTLENECK', (4, ),        (64, )],
                [1, 2, 'BASIC',      (4, 4),       (64, 128)],
                [4, 3, 'BASIC',      (4, 4, 4),    (64, 128, 256)],
                [3, 4, 'BASIC',      (4, 4, 4, 4), (64, 128, 256, 512)]],
    }  # yapf:disable

    def __init__(self,
                 arch='w32',
                 extra=None,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 multiscale_output=True,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(HRNet, self).__init__(init_cfg)

        extra = self.parse_arch(arch, extra)

        # Assert configurations of 4 stages are in extra
        for i in range(1, 5):
            assert f'stage{i}' in extra, f'Missing stage{i} config in "extra".'
            # Assert whether the length of `num_blocks` and `num_channels` are
            # equal to `num_branches`
            cfg = extra[f'stage{i}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # -------------------- stem net --------------------
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # -------------------- stage 1 --------------------
        self.stage1_cfg = self.extra['stage1']
        base_channels = self.stage1_cfg['num_channels']
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in base_channels
        ]
        # To align with the original code, use layer1 instead of stage1 here.
        self.layer1 = ResLayer(
            block,
            in_channels=64,
            out_channels=num_channels[0],
            num_blocks=num_blocks[0])
        pre_num_channels = num_channels

        # -------------------- stage 2~4 --------------------
        for i in range(2, 5):
            stage_cfg = self.extra[f'stage{i}']
            base_channels = stage_cfg['num_channels']
            block = self.blocks_dict[stage_cfg['block']]
            multiscale_output_ = multiscale_output if i == 4 else True

            num_channels = [
                channel * get_expansion(block) for channel in base_channels
            ]
            # The transition layer from layer1 to stage2
            transition = self._make_transition_layer(pre_num_channels,
                                                     num_channels)
            self.add_module(f'transition{i-1}', transition)
            stage = self._make_stage(
                stage_cfg, num_channels, multiscale_output=multiscale_output_)
            self.add_module(f'stage{i}', stage)

            pre_num_channels = num_channels

        self.cbam = CBAM(in_channels=num_channels[0])

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # For existing scale branches,
                # add conv block when the channels are not the same.
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                # For new scale branches, add stacked downsample conv blocks.
                # For example, num_branches_pre = 2, for the 4th branch, add
                # stacked two downsample conv blocks.
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        block_init_cfg = None
        if self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))

        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg))

        return Sequential(*hr_modules)

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = [x]

        for i in range(2, 5):
            # Apply transition
            transition = getattr(self, f'transition{i-1}')
            inputs = []
            for j, layer in enumerate(transition):
                if j < len(x_list):
                    inputs.append(layer(x_list[j]))
                else:
                    inputs.append(layer(x_list[-1]))
            # Forward HRModule
            stage = getattr(self, f'stage{i}')
            x_list = stage(inputs)

        residual = x_list[0].clone()
        cbam_branch = self.cbam(x_list[0])
        x_list[0] = residual + cbam_branch


        return tuple(x_list)

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(HRNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def parse_arch(self, arch, extra=None):
        if extra is not None:
            return extra

        assert arch in self.arch_zoo, \
            ('Invalid arch, please choose arch from '
             f'{list(self.arch_zoo.keys())}, or specify `extra` '
             'argument directly.')

        extra = dict()
        for i, stage_setting in enumerate(self.arch_zoo[arch], start=1):
            extra[f'stage{i}'] = dict(
                num_modules=stage_setting[0],
                num_branches=stage_setting[1],
                block=stage_setting[2],
                num_blocks=stage_setting[3],
                num_channels=stage_setting[4],
            )

        return extra
