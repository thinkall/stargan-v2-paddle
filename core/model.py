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


class LeakyRelu(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.leaky_relu = lambda x: fluid.layers.leaky_relu(x, alpha=alpha)

    def forward(self, x):
        return self.leaky_relu(x)


class ResBlk(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        """

        :param dim_in: in_channels
        :param dim_out: out_channels
        :param normalize:
        :param downsample:
        """
        super(ResBlk, self).__init__()
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.avg_pool2d = fluid.dygraph.Pool2D(pool_size=2, pool_stride=2, pool_padding=0, pool_type='avg')
        self.actv = LeakyRelu(0.2)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = fluid.dygraph.InstanceNorm(
                dim_in, epsilon=1e-05,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0), trainable=False),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0), trainable=False),
                dtype='float32')  # affine=False,对应代码中的两个参数设置
            self.norm2 = fluid.dygraph.InstanceNorm(
                dim_in, epsilon=1e-05,
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0), trainable=False),
                bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0), trainable=False),
                dtype='float32')  # affine=False,对应代码中的两个参数设置
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
        self.norm = fluid.dygraph.InstanceNorm(
            num_features, epsilon=1e-05,
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0), trainable=False),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0), trainable=False),
            dtype='float32')  # affine=False,对应代码中的两个参数设置
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        # print(f's.shape = {s.shape}')
        h = self.fc(s)  # h: [batch_size, num_features * 2]
        # print(f'good s.shape = {s.shape}')
        h = fluid.layers.reshape(h, shape=[h.shape[0], h.shape[1], 1, 1])
        h = fluid.layers.split(h, num_or_sections=2, dim=1)
        gamma, beta = h[0], h[1]
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = LeakyRelu(0.2)
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
        filter = fluid.dygraph.to_variable(self.filter)
        filter = fluid.layers.concat([filter] * x.shape[1], axis=0)
        # return paddle.nn.functional.conv2d(x, filter, padding=1, groups=x.shape[1])  # need paddle develop version
        return conv2d_with_filter(x, filter, padding=1, groups=x.shape[1])


class Generator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size  # dim_in: 64
        self.img_size = img_size
        self.from_rgb = nn.Conv2D(3, dim_in, 3, 1, 1)
        self.encode = fluid.dygraph.Sequential()
        self.decode = fluid.dygraph.Sequential()
        self.to_rgb = fluid.dygraph.Sequential(fluid.dygraph.InstanceNorm(
                                                   dim_in, epsilon=1e-05,
                                                   param_attr=fluid.ParamAttr(
                                                       initializer=fluid.initializer.Constant(1.0), trainable=False),
                                                   bias_attr=fluid.ParamAttr(
                                                       initializer=fluid.initializer.Constant(0.0), trainable=False),
                                                   dtype='float32'),  # affine=False,对应代码中的两个参数设置
                                               # functools.partial(fluid.layers.leaky_relu, alpha=0.2),
                                               LeakyRelu(alpha=0.2),
                                               nn.Conv2D(dim_in, 3, 1, 1, 0))
        self.w_hpf = w_hpf

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4  # repeat_num: 4
        self.repeat_num = repeat_num
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.add_sublayer(f'lsample_{_}',
                                     ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.add_sublayer(f'lsample_{_}',
                                     AdainResBlk(dim_out, dim_in, style_dim,
                                                 w_hpf=w_hpf, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.add_sublayer(f'lbnk_{_}', ResBlk(dim_out, dim_out, normalize=True))
            self.decode.add_sublayer(f'lbnk_{_}', AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        self.decode1 = fluid.dygraph.Sequential()
        for _ in list(range(2))[::-1]:
            layer = self.decode[f'lbnk_{_}']
            self.decode1.add_sublayer(f'lbnk_{_}', layer)
        for _ in list(range(repeat_num))[::-1]:
            layer = self.decode[f'lsample_{_}']
            self.decode1.add_sublayer(f'lsample_{_}', layer)
        self.decode = self.decode1  # stack-like

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}

        for _ in range(self.repeat_num):
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = self.encode[f'lsample_{_}'](x)

        for _ in range(2):
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = self.encode[f'lbnk_{_}'](x)

        for _ in list(range(2))[::-1]:
            x = self.decode[f'lbnk_{_}'](x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = fluid.layers.interpolate(mask, out_shape=x.shape[2], resample='BILINEAR')
                if self.w_hpf > 0:
                    x = x + self.hpf(mask * cache[x.shape[2]])

        for _ in list(range(self.repeat_num))[::-1]:
            x = self.decode[f'lsample_{_}'](x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = fluid.layers.interpolate(mask, out_shape=x.shape[2], resample='BILINEAR')
                if self.w_hpf > 0:
                    x = x + self.hpf(mask * cache[x.shape[2]])

        return self.to_rgb(x)


class MappingNetwork(fluid.dygraph.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        self.num_domains = num_domains
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
        for _ in range(self.num_domains):
            out += [self.unshared[f'lsub_{_}'](h)]
        out = paddle.fluid.layers.stack(out, axis=2)  # (batch, style_dim, num_domains)
        out = fluid.layers.transpose(out, [0, 2, 1])  # (batch, num_domains, style_dim)
        idx = np.array(range(y.shape[0]))
        s = out.numpy()[idx, y.numpy().astype('int')]  # (batch, style_dim)
        return fluid.dygraph.to_variable(s)


class StyleEncoder(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.num_domains = num_domains
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [LeakyRelu(alpha=0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [LeakyRelu(alpha=0.2)]
        self.shared = fluid.dygraph.Sequential(*blocks)

        self.unshared = fluid.dygraph.Sequential()
        for _ in range(num_domains):
            self.unshared.add_sublayer(f'lsub_{_}', nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        h = self.shared(x)
        h = fluid.layers.reshape(h, shape=[h.shape[0], -1])
        out = []
        for _ in range(self.num_domains):
            out += [self.unshared[f'lsub_{_}'](h)]
        out = paddle.fluid.layers.stack(out, axis=2)  # (batch, style_dim, num_domains)
        out = fluid.layers.transpose(out, [0, 2, 1])  # (batch, num_domains, style_dim)
        idx = np.array(range(y.shape[0]))
        s = out.numpy()[idx, y.numpy()]  # (batch, style_dim)
        return fluid.dygraph.to_variable(s)


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [LeakyRelu(alpha=0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 4, 1, 0)]
        blocks += [LeakyRelu(alpha=0.2)]
        blocks += [nn.Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = fluid.dygraph.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])  # (batch, num_domains)
        idx = np.array(range(y.shape[0]))
        out = out.numpy()[idx, y.numpy()]  # (batch)
        return fluid.dygraph.to_variable(out)


def soft_update(source, target, decay=1.0):
    assert 0.0 <= decay <= 1.0
    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(decay * source_param + (1.0 - decay) * target_param)


def build_model(args):
    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)
    generator_ema = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network_ema = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder_ema = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    soft_update(generator, generator_ema)
    soft_update(mapping_network, mapping_network_ema)
    soft_update(style_encoder, style_encoder_ema)

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    args = parser.parse_args()
    place = paddle.fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        nets, nets_ema = build_model(args)
        print(nets)
        print(nets_ema)
        print(nets.generator.state_dict().keys())
        assert (nets.generator.state_dict()['from_rgb.weight'].numpy()
                == nets_ema.generator.state_dict()['from_rgb.weight'].numpy()).all()
    print(nets.keys(), nets_ema.keys())
    for k in nets:
        print(k)

