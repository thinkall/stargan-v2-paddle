import copy
import math

from munch import Munch
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Conv2DTranspose, InstanceNorm, PRelu

from core.wing import FAN


class ReLU(fluid.Layer):
    def __init__(self):
        super().__init__()
        self.relu = lambda x: fluid.layers.relu(x)

    def forward(self, x):
        return self.relu(x)


class ResBlk(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, actv='leaky_relu', normalize=False, downsample=False):
        super(ResBlk, self).__init__()
        self.normalize = normalize
        self.downsample = downsample
        self.actv = actv
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = InstanceNorm(dim_in, epsilon=1e-05,
                                      param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0),
                                                                 trainable=True),
                                      bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                                trainable=True),
                                      dtype='float32')  # affine=True,对应代码中的两个参数设置
            self.norm2 = InstanceNorm(dim_in, epsilon=1e-05,
                                      param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0),
                                                                 trainable=True),
                                      bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                                trainable=True),
                                      dtype='float32')  # affine=True,对应代码中的两个参数设置
        if self.learned_sc:
            self.conv1x1 = Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = fluid.layers.pool2d(x, pool_size=2, pool_type="avg", pool_stride=2, pool_padding=0)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        if self.actv == 'leaky_relu':
            x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv1(x)
        if self.downsample:
            x = fluid.layers.pool2d(x, pool_size=2, pool_type="avg", pool_stride=2, pool_padding=0)
        if self.normalize:
            x = self.norm2(x)
        if self.actv == 'leaky_relu':
            x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(fluid.dygraph.Layer):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = InstanceNorm(num_features, epsilon=1e-05,
                                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0),
                                                            trainable=False),
                                 bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                           trainable=False),
                                 dtype='float32')  # affine=False,对应代码中的两个参数设置
        self.fc = Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = fluid.layers.reshape(h, shape=[h.shape[0], h.shape[1], 1, 1])
        #   print(h.shape) [8, 1024, 1, 1]
        #  gamma, beta=fluid.layers.unstack(h, axis=1, num=2)
        gamma, beta = fluid.layers.split(h, num_or_sections=2, dim=1)
        #    print(gamma.shape) [8, 512, 1, 1]
        #    print(self.norm(x).shape)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv='leaky_relu', upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.upsample:
            x = fluid.layers.image_resize(x, scale=2, resample="NEAREST")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        if self.actv == 'leaky_relu':
            x = fluid.layers.leaky_relu(x, alpha=0.2)
        if self.upsample:
            x = fluid.layers.image_resize(x, scale=2, resample="NEAREST")
        x = self.conv1(x)
        x = self.norm2(x, s)
        if self.actv == 'leaky_relu':
            x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            #  print(out.shape,self._shortcut(x).shape)
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(fluid.dygraph.Layer):
    def __init__(self, w_hpf, x):  # 传个x进去好知道维度
        super(HighPass, self).__init__()
        self.filter = fluid.dygraph.to_variable(np.array([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]])) / w_hpf
        self.filter = fluid.layers.unsqueeze(input=self.filter, axes=[0])
        self.filter = fluid.layers.unsqueeze(input=self.filter, axes=[1])
        self.filter = fluid.layers.expand(self.filter, expand_times=[x.shape[1], 1, 1,
                                                                     1])  # 滤波器的维度是 [M, C, H, W] ，M是输出特征图个数，C是输入特征图个数，如果组数大于1，C等于输入特征图个数除以组数的结果。
        self.filter = self.filter.numpy()
        self.conv2 = Conv2D(x.shape[1], x.shape[1], 3, stride=1, padding=1, groups=x.shape[1],
                            param_attr=fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(self.filter),
                                                       trainable=False))  # 维度可能有误，待定

    def forward(self, x):
        return self.conv2(x)


class Generator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = Conv2D(3, dim_in, 3, 1, 1)
        self.encode = list()
        self.decode = list()
        self.norm1 = InstanceNorm(dim_in, epsilon=1e-05,
                                  param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0),
                                                             trainable=True),
                                  bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                                            trainable=True),
                                  dtype='float32')  # affine=True,对应代码中的两个参数设置
        self.conv1 = Conv2D(dim_in, 3, 1, 1, 0)
        self.w_hpf = w_hpf

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

    def to_rgb(self, x):
        x = self.norm1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.conv1(x)
        return x

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
                mask = fluid.layers.image_resize(mask, out_shape=[x.shape[2], x.shape[2]], resample="BILINEAR")
                if self.w_hpf > 0:
                    self.hpf = HighPass(self.w_hpf, mask * cache[x.shape[2]])
                # print(x.shape,self.hpf(mask * cache[x.shape[2]]).shape)   匹配
                x = x + self.hpf(mask * cache[x.shape[2]])
        return self.to_rgb(x)


class MappingNetwork(fluid.dygraph.Layer):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        self.fc1 = Linear(latent_dim, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 512)
        self.fc4 = Linear(512, 512)

        self.unshared = list()
        for _ in range(num_domains):
            self.unshared += [fluid.dygraph.Sequential(Linear(512, 512),
                                                       ReLU(),
                                                       Linear(512, 512),
                                                       ReLU(),
                                                       Linear(512, 512),
                                                       ReLU(),
                                                       Linear(512, style_dim))]

    # print(self.unshared,1)
    # self.unshared=fluid.dygraph.LayerList(self.unshared)
    # print(self.unshared,2)
    def shared(self, x):
        x = self.fc1(x)
        x = fluid.layers.relu(x)
        x = self.fc2(x)
        x = fluid.layers.relu(x)
        x = self.fc3(x)
        x = fluid.layers.relu(x)
        x = self.fc4(x)
        x = fluid.layers.relu(x)
        return x

    def forward(self, z, y):
        #      print(type(z))
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        #   print(out)
        out = fluid.layers.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = np.array(range(y.shape[0]))
        out = out.numpy()
        y = y.numpy()
        s = out[idx, y]  # (batch, style_dim)
        s = fluid.dygraph.to_variable(s)
        return s


class StyleEncoder(fluid.dygraph.Layer):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.shared_up = fluid.dygraph.Sequential(*blocks)
        self.blocks1 = Conv2D(dim_out, dim_out, 4, 1, 0)

        self.unshared = list()
        for _ in range(num_domains):
            self.unshared += [Linear(dim_out, style_dim)]

    def shared(self, x):
        x = self.shared_up(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.blocks1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2)
        return x

    def forward(self, x, y):
        h = self.shared(x)
        h = fluid.layers.reshape(h, shape=[h.shape[0], -1])
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = fluid.layers.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = np.array(range(y.shape[0]))
        out = out.numpy()
        y = y.numpy()
        s = out[idx, y]  # (batch, style_dim)
        s = fluid.dygraph.to_variable(s)
        return s


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [Conv2D(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        self.main_up = fluid.dygraph.Sequential(*blocks)

        self.blocks1 = Conv2D(dim_out, dim_out, 4, 1, 0)
        self.blocks2 = Conv2D(dim_out, num_domains, 1, 1, 0)

    def main(self, x):
        x = self.main_up(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.blocks1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2)
        x = self.blocks2(x)
        return x

    def forward(self, x, y):
        out = self.main(x)
        out = fluid.layers.reshape(out, shape=[out.shape[0], -1])
        idx = np.array(range(y.shape[0]))
        #    print(type(out))  <class 'paddle.fluid.core_avx.VarBase'>
        out = out.numpy()
        y = y.numpy()
        s = out[idx, y]  # (batch, style_dim)
        s = fluid.dygraph.to_variable(s)
        return s


def copy_from(target, source):
    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(source_param)


def build_model(args):
    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.img_size, args.num_domains)

    generator_ema = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    mapping_network_ema = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder_ema = StyleEncoder(args.img_size, args.style_dim, args.num_domains)

    copy_from(generator_ema, generator)
    copy_from(mapping_network_ema, mapping_network)
    copy_from(style_encoder_ema, style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path)
        fan.eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema