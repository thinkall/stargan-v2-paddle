# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import paddle
import paddle.fluid as fluid

from core.model import build_model, soft_update
from core.checkpoint import CheckpointIO
from core.data_reader import InputFetcher
import core.utils as utils


class Solver(object):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            place = paddle.fluid.CUDAPlace(self.args.whichgpu) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                self.optims = Munch()
                self.ckptios = Munch()
                for net in self.nets.keys():
                    if net == 'fan':
                        continue
                    self.optims[net] = fluid.optimizer.AdamOptimizer(
                        learning_rate=args.f_lr if net == 'mapping_network' else args.lr,
                        beta1=args.beta1,
                        beta2=args.beta2,
                        parameter_list=self.nets[net].parameters(),
                        regularization=fluid.regularizer.L2Decay(
                            regularization_coeff=args.weight_decay)
                    )
                    self.ckptios[net] = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema_' + net)),
                                         CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_' + net))]
        else:
            self.ckptios = Munch()
            for net in self.nets.keys():
                self.ckptios[net] = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema_' + net))]

        # todo: add kaiming initializing

    def _save_checkpoint(self, step):
        for net in self.ckptios:
            try:
                if len(self.ckptios[net]) == 1:
                    self.ckptios[net][0].save(step, self.nets_ema[net], 'para')
                    # self.ckptios[net][0].save(step, self.optims[net], 'opti')
                else:
                    self.ckptios[net][0].save(step, self.nets_ema[net], 'para')
                    # self.ckptios[net][0].save(step, self.optims[net], 'opti')
                    self.ckptios[net][1].save(step, self.nets[net], 'para')
                    self.ckptios[net][1].save(step, self.optims[net], 'opti')
            except Exception as e:
                print(e)

    def _load_checkpoint(self, step):
        for net in self.ckptios:
            if len(self.ckptios[net]) == 1:
                para_state_dict, opti_state_dict = self.ckptios[net][0].load(step)
                self.nets_ema[net].load_dict(para_state_dict)
            else:
                para_state_dict, opti_state_dict = self.ckptios[net][0].load(step)
                self.nets_ema[net].load_dict(para_state_dict)
                para_state_dict, opti_state_dict = self.ckptios[net][1].load(step)
                self.nets[net].load_dict(para_state_dict)
                self.optims[net].load_dict(opti_state_dict)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.clear_gradients()

    def train(self, loaders):
        place = paddle.fluid.CUDAPlace(self.args.whichgpu) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            args = self.args
            nets = self.nets
            nets_ema = self.nets_ema
            optims = self.optims

            # fetch random validation images for debugging
            fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
            fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
            inputs_val = next(fetcher_val)

            # resume training if necessary
            if args.resume_iter > 0:
                self._load_checkpoint(args.resume_iter)

            # remember the initial value of ds weight
            initial_lambda_ds = args.lambda_ds

            print('Start training...')
            start_time = time.time()
            for i in range(args.resume_iter, args.total_iters):
                # fetch images and labels
                inputs = next(fetcher)
                x_real, y_org = inputs.x_src, inputs.y_src
                x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
                z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
                x_real = fluid.dygraph.to_variable(x_real.astype('float32'))
                y_org = fluid.dygraph.to_variable(y_org)
                x_ref = fluid.dygraph.to_variable(x_ref.astype('float32'))
                x_ref2 = fluid.dygraph.to_variable(x_ref2.astype('float32'))
                y_trg = fluid.dygraph.to_variable(y_trg)
                z_trg = fluid.dygraph.to_variable(z_trg.astype('float32'))
                z_trg2 = fluid.dygraph.to_variable(z_trg2.astype('float32'))

                masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

                # train the discriminator
                # print('train the discriminator')
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self._reset_grad()
                d_loss_avg = fluid.layers.mean(d_loss)
                d_loss_avg.backward()
                optims.discriminator.minimize(d_loss_avg)

                d_loss, d_losses_ref = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
                self._reset_grad()
                d_loss_avg = fluid.layers.mean(d_loss)
                d_loss_avg.backward()
                optims.discriminator.minimize(d_loss_avg)

                # train the generator
                # print('train the generator 1st')
                g_loss, g_losses_latent = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self._reset_grad()
                g_loss_avg = fluid.layers.mean(g_loss)
                g_loss_avg.backward()  # stuck here with 1.8.x cpu version
                optims.generator.minimize(g_loss_avg)
                optims.mapping_network.minimize(g_loss_avg)
                optims.style_encoder.minimize(g_loss_avg)

                # print('train the generator 2nd')

                g_loss, g_losses_ref = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
                self._reset_grad()
                g_loss_avg = fluid.layers.mean(g_loss)
                g_loss_avg.backward()
                optims.generator.minimize(g_loss_avg)

                # compute moving average of network parameters
                # print('compute moving average of network parameters')
                moving_average(nets.generator, nets_ema.generator, beta=0.999)
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

                # print('finish compute moving average of network parameters')

                # decay weight for diversity sensitive loss
                if args.lambda_ds > 0:
                    args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

                # print out log info
                if (i+1) % args.print_every == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                    log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                    all_losses = dict()
                    for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    all_losses['G/lambda_ds'] = args.lambda_ds
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                    print(log)

                # generate images for debugging
                if (i+1) % args.sample_every == 0:
                    os.makedirs(args.sample_dir, exist_ok=True)
                    utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

                # save model checkpoints
                if (i+1) % args.save_every == 0:
                    self._save_checkpoint(step=i+1)

                # # compute FID and LPIPS if necessary
                # if (i+1) % args.eval_every == 0:
                #     calculate_metrics(nets_ema, args, i+1, mode='latent')
                #     calculate_metrics(nets_ema, args, i+1, mode='reference')

    @fluid.dygraph.no_grad
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

    @fluid.dygraph.no_grad
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        # calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.stop_gradient = False
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = 0  # r1_reg(out, x_real)

    # with fake images
    with fluid.dygraph.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.numpy(),
                       fake=loss_fake.numpy(),
                       reg=loss_reg)


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = fluid.layers.mean(fluid.layers.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = fluid.layers.mean(fluid.layers.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = fluid.layers.mean(fluid.layers.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.numpy(),
                       sty=loss_sty.numpy(),
                       ds=loss_ds.numpy(),
                       cyc=loss_cyc.numpy())


def moving_average(model, model_test, beta=0.999):
    soft_update(model, model_test, decay=beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = fluid.layers.ones_like(logits) * target
    loss = fluid.layers.softmax_with_cross_entropy(logits, targets, soft_label=True)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    d_out_sum = fluid.layers.sum(d_out)
    grad_dout = fluid.dygraph.grad(
        outputs=d_out_sum, inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.shape == x_in.shape)
    grad_dout3 = fluid.layers.reshape(grad_dout2, shape=[batch_size, -1])
    reg = 0.5 * fluid.layers.reduce_sum(grad_dout3, dim=1)
    return reg
