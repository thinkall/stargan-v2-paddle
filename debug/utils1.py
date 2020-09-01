import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile
import cv2
from tqdm import tqdm
import ffmpeg

import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.utils as vutils
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Conv2DTranspose, InstanceNorm, PRelu


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):  # 未转成paddlepaddle
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):  # 不行只能一一设置弄过去
    if isinstance(module, Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    out = fluid.layers.clamp(out, min=0, max=1)
    return out


@fluid.dygraph.no_grad
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    x_src = fluid.dygraph.to_variable(x_src.numpy())
    y_src = fluid.dygraph.to_variable(y_src.numpy())
    x_ref = fluid.dygraph.to_variable(x_ref.numpy())
    y_ref = fluid.dygraph.to_variable(y_ref.numpy())

    N, C, H, W = x_src.shape
    # print(N) 32
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]  # 32*4=128
    x_concat = fluid.layers.concat(x_concat, axis=0)

    save_image(x_concat, N, filename)
    del x_concat


@fluid.dygraph.no_grad
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    x_src = fluid.dygraph.to_variable(x_src.numpy())
    N, C, H, W = x_src.shape
    latent_dim = z_trg_list[0].shape[1]
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):

        z_many = fluid.dygraph.to_variable(np.random.randn(10000, latent_dim))
        z_many = fluid.layers.cast(x=z_many, dtype=np.float32)
        y_many = fluid.layers.ones(shape=[10000], dtype='int32') * y_trg[0].numpy()[0]
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = fluid.layers.reduce_mean(s_many, dim=0, keep_dim=True)
        s_avg = fluid.layers.expand(s_avg, expand_times=[N, 1])
        # print(z_trg_list.shape) [10, 32, 16]
        for z_trg in z_trg_list.numpy():
            z_trg = fluid.dygraph.to_variable(z_trg)
            z_trg = fluid.layers.cast(x=z_trg, dtype=np.float32)
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = (s_trg - s_avg) * psi + s_avg
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N, filename)


@fluid.dygraph.no_grad
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    x_src = fluid.dygraph.to_variable(x_src.numpy())
    y_ref = fluid.dygraph.to_variable(y_ref.numpy())
    x_ref = fluid.dygraph.to_variable(x_ref.numpy())
    N, C, H, W = x_src.shape
    wb = fluid.layers.ones(shape=[1, C, H, W], dtype='float32')
    x_src_with_wb = fluid.layers.concat([wb, x_src], axis=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_ref = fluid.layers.unsqueeze(input=s_ref, axes=[1])
    s_ref_list = fluid.layers.expand(s_ref, expand_times=[1, N, 1])
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list.numpy()):
        s_ref = fluid.dygraph.to_variable(s_ref)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = fluid.layers.concat([x_ref[i:i + 1], x_fake], axis=0)
        x_concat += [x_fake_with_ref]
    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N + 1, filename)
    del x_concat


@fluid.dygraph.no_grad
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    N = inputs.x_src.shape[0]
    # print(N) 32
    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [fluid.layers.expand(fluid.dygraph.to_variable(np.array([y])), expand_times=[N])
                  for y in range(min(args.num_domains, 5))]

    z_trg_list = fluid.layers.expand(
        fluid.dygraph.to_variable(np.random.randn(args.num_outs_per_domain, 1, args.latent_dim)),
        expand_times=[1, N, 1])

    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

        # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #

def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.shape[0]
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = (s_next - s_prev) * alpha + s_prev
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = fluid.layers.concat([x_src, x_fake], axis=2)

        frame = make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frame = fluid.layers.unsqueeze(input=frame, axes=[0])
        frames.append(frame)
    frames = fluid.layers.concat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next,
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas)  # number of frames
    canvas = - fluid.layers.ones(shape=[T, C, H * 2, W + margin], dtype='float32')
    merged = fluid.layers.concat(entries, axis=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]  # 可能需要先转成numpy
    return canvas


# 这里有三个video相关的函数不考虑


def tensor2ndarray255(images):
    images = fluid.layers.clamp(images * 0.5 + 0.5, min=0, max=1)
    return images.numpy().transpose(0, 2, 3, 1) * 255


from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib

import math


def make_grid(
        tensor,
        nrow: int = 8,
        padding: int = 2,
        pad_value: int = 0,
):
    if len(tensor.shape) == 2:  # single image H x W
        tensor = fluid.layers.unsqueeze(input=tensor, axes=[0])
    if len(tensor.shape) == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = fluid.layers.concat(input=[tensor, tensor, tensor], axis=0)
        tensor = fluid.layers.unsqueeze(input=tensor, axes=[0])

    if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = fluid.layers.concat(input=[tensor, tensor, tensor], axis=1)

    if tensor.shape[1] == 1:
        return fluid.layers.squeeze(input=tensor, axes=[0])

        # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    #   print(height,width) 258 258
    num_channels = tensor.shape[1]
    # grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    data = fluid.dygraph.to_variable(np.zeros((num_channels, height * ymaps + padding, width * xmaps + padding)))
    grid = fluid.layers.full_like(data, pad_value)
    #  print(grid.shape) [3, 1034, 8258]
    k = 0
    grid = grid.numpy()
    tensor = tensor.numpy()
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            try:
                grid[:, y * height:(y + 1) * height - padding, x * width:(x + 1) * width - padding] = tensor[k]
            except ValueError as e:
                print(str(e))
                print(k)
            k = k + 1
    return grid


def save_image(
        tensor,
        nrow,
        fp,
        padding: int = 2,
        pad_value: int = 0
):
    tensor = denormalize(tensor)
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.transpose(np.clip(grid * (255) + (0.5), 0, 255), (1, 2, 0))
    im = Image.fromarray(np.uint8(ndarr))
    im.save(fp, )

