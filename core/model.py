# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import copy
import math
import functools
from munch import Munch
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn

from core.base_network import conv2d_with_filter


class ResBlk(fluid.dygraph.Layer):
    def __init__(self, name_scope, dim_in, dim_out, normalize=False, downsample=False):
        """

        :param name_scope:
        :param dim_in: in_channels
        :param dim_out: out_channels
        :param normalize:
        :param downsample:
        """
        super(ResBlk, self).__init__(name_scope)
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.avg_pool2d = fluid.dygraph.Pool2D(pool_size=2, pool_stride=2, pool_padding=0, pool_type='avg')
        self.actv = functools.partial(fluid.layers.leaky_relu, alpha=0.2)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = fluid.dygraph.InstanceNorm(dim_in)
            self.norm2 = fluid.dygraph.InstanceNorm(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.avg_pool2d(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = self.avg_pool2d(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(fluid.dygraph.Layer):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = fluid.dygraph.InstanceNorm(num_features)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = fluid.layers.reshape(h, shape=[h.shape[0], h.shape[1], 1, 1])
        h = fluid.layers.unstack(h, axis=2)
        gamma, beta = h[0], h[1]
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = functools.partial(fluid.layers.leaky_relu, alpha=0.2)
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.upsample:
            x = fluid.layers.interpolate(x, scale=2, resample='NEAREST')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = fluid.layers.interpolate(x, scale=2, resample='NEAREST')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(fluid.dygraph.Layer):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        tmp = np.array([[-1, -1, -1],
                        [-1, 8., -1],
                        [-1, -1, -1]])
        tmp = np.expand_dims(np.expand_dims(tmp, axis=0), axis=1)
        self.filter = tmp / w_hpf

    def forward(self, x):
        filter = self.filter.repeat(x.shape[1], 0)
        filter = fluid.dygraph.to_variable(filter)
        # return paddle.nn.functional.conv2d(x, filter, padding=1, groups=x.shape[1])  # need paddle develop version
        return conv2d_with_filter(x, filter, padding=1, groups=x.shape[1])


class Generator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2D(3, dim_in, 3, 1, 1)
        self.encode = fluid.dygraph.Sequential()
        self.decode = fluid.dygraph.Sequential()
        self.to_rgb = fluid.dygraph.Sequential(fluid.dygraph.InstanceNorm(dim_in),
                                               functools.partial(fluid.layers.leaky_relu, alpha=0.2),
                                               nn.Conv2D(dim_in, 3, 1, 1, 0))
        self.w_hpf = w_hpf

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.add_sublayer(f'l{_}',
                                     ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.add_sublayer(f'l{_}',
                                     AdainResBlk(dim_out, dim_in, style_dim,
                                                 w_hpf=w_hpf, upsample=True))
            dim_in = dim_out
        self.decode1 = fluid.dygraph.Sequential()
        i = 0
        for layer in self.decode.sublayers()[::-1]:
            i += 1
            self.decode1.add_sublayer(f'l{i}', layer)
        self.decode = self.decode1  # stack-like

        # bottleneck blocks
        for _ in range(2):
            self.encode.add_sublayer(f'l{_}', ResBlk(dim_out, dim_out, normalize=True))
            self.decode.add_sublayer(f'l{_}', AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = fluid.layers.interpolate(mask, out_shape=x.shape[2], resample='BILINEAR')
                if self.w_hpf > 0:
                    x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)


class MappingNetwork(fluid.dygraph.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512, act='relu')]
        for _ in range(3):
            layers += [nn.Linear(512, 512, act='relu')]
        self.shared = fluid.dygraph.Sequential(*layers)

        self.unshared = fluid.dygraph.Sequential()
        for _ in range(num_domains):
            sublayer = fluid.dygraph.Sequential(nn.Linear(512, 512, act='relu'),
                                                nn.Linear(512, 512, act='relu'),
                                                nn.Linear(512, 512, act='relu'),
                                                nn.Linear(512, style_dim))
            self.unshared.add_sublayer(f'lsub_{_}', sublayer)

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.fluid.layers.stack(out, axis=2)  # (batch, num_domains, style_dim)
        idx = fluid.dygraph.to_variable(np.array(range(y.shape[0])))
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.leaky_relu = functools.partial(fluid.layers.leaky_relu, alpha=0.2)
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [self.leaky_relu]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [self.leaky_relu]
        self.shared = fluid.dygraph.Sequential(*blocks)

        self.unshared = fluid.dygraph.Sequential()
        for _ in range(num_domains):
            self.unshared.add_sublayer(f'lsub_{_}', nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        h = self.shared(x)
        h = fluid.layers.reshape(h, shape=[h.shape[0], -1])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.fluid.layers.stack(out, axis=2)   # (batch, num_domains, style_dim)
        idx = fluid.dygraph.to_variable(np.array(range(y.shape[0])))
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.leaky_relu = functools.partial(fluid.layers.leaky_relu, alpha=0.2)
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [self.leaky_relu]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [self.leaky_relu]
        blocks += [nn.Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = fluid.dygraph.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])  # (batch, num_domains)
        idx = fluid.dygraph.to_variable(np.array(range(y.shape[0])))
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = FAN(fname_pretrained=args.wing_path).eval()
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema