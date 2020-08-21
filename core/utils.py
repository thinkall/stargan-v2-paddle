# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import os
from os.path import join as ospj
import numpy as np
from PIL import Image
import cv2

import paddle
import paddle.fluid as fluid


def denormalize(x):
    out = (x + 1) / 2
    return fluid.layers.clamp(out, min=0, max=1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    x = x * 255 + 0.5
    # print(x.shape, ncol, filename)
    N, C, H, W = x.shape
    nrow = N // ncol
    img = np.zeros((H * nrow, W * ncol, 3))
    # print(img.shape)
    for idx in range(N):
        i = idx % ncol
        j = idx // ncol
        img[H*j:H*(j+1), W*i:W*(i+1), :] = x[idx].numpy().transpose(1,2,0)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
    # print(f'good for {filename}')
    return cv2.imwrite(filename, img)


def torch_lerp(start, end, weight):
    out = fluid.layers.elementwise_add(start, weight * fluid.layers.elementwise_sub(end, start))
    return out


@fluid.dygraph.no_grad
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N, filename)
    del x_concat


@fluid.dygraph.no_grad
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, C, H, W = x_src.shape
    latent_dim = z_trg_list[0].shape[1]
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = np.random.randn(10000, latent_dim)
        y_many = np.ones(10000) * y_trg[0].numpy()
        z_many = fluid.dygraph.to_variable(z_many.astype('float32'))
        y_many = fluid.dygraph.to_variable(y_many)
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = fluid.layers.reduce_mean(s_many, dim=0, keep_dim=True)
        s_avg = fluid.layers.concat([s_avg] * N, axis=0)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg.astype('float32'), y_trg)
            s_trg = torch_lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N, filename)


@fluid.dygraph.no_grad
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    N, C, H, W = x_src.shape
    wb = np.ones([1, C, H, W])
    wb = fluid.dygraph.to_variable(wb.astype('float32'))
    x_src_with_wb = fluid.layers.concat([wb, x_src], axis=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    x_concat = [x_src_with_wb]
    for i in range(N):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = fluid.layers.concat([x_ref[i:i+1], x_fake], axis=0)
        x_concat += [x_fake_with_ref]

    x_concat = fluid.layers.concat(x_concat, axis=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@fluid.dygraph.no_grad
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref
    x_src = fluid.dygraph.to_variable(x_src.astype('float32'))
    y_src = fluid.dygraph.to_variable(y_src)
    x_ref = fluid.dygraph.to_variable(x_ref.astype('float32'))
    y_ref = fluid.dygraph.to_variable(y_ref)

    N = inputs.x_src.shape[0]

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = np.asarray([np.array(y).repeat(N) for y in range(min(args.num_domains, 5))])
    z_trg_list = np.random.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(N, axis=1)
    y_trg_list = fluid.dygraph.to_variable(y_trg_list)
    z_trg_list = fluid.dygraph.to_variable(z_trg_list)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)
