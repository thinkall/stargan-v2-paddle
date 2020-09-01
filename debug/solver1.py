import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Conv2DTranspose, InstanceNorm, PRelu

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils


# from metrics.eval import calculate_metrics

class Solver(fluid.dygraph.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            #    utils.print_network(module, name) 未有对应实现
            #      print(name,module)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            #    print(name,module)
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = fluid.optimizer.AdamOptimizer(
                    learning_rate=args.f_lr if net == 'mapping_network' else args.lr, beta1=args.beta1,
                    beta2=args.beta2, parameter_list=self.nets[net].parameters(),
                    regularization=fluid.regularizer.L2Decay(regularization_coeff=args.weight_decay))

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims'), **self.optims)]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema'), **self.nets_ema)]

        # for name, network in self.named_children(): #直接找对应的初始化得了
        #     # Do not initialize the FAN parameters
        #     if ('ema' not in name) and ('fan' not in name):
        #         print('Initializing %s...' % name)
        #         network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            #   optim.zero_grad()
            optim.clear_gradients()

    def train(self, loaders):

        # self.nets.fan.eval()

        # self.nets.style_encoder.train()
        # self.nets.discriminator.train()
        # self.nets.mapping_network.train()
        # self.nets.generator.train()

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

            x_real = fluid.dygraph.to_variable(x_real.numpy())
            y_org = fluid.dygraph.to_variable(y_org.numpy())
            x_ref = fluid.dygraph.to_variable(x_ref.numpy())
            x_ref2 = fluid.dygraph.to_variable(x_ref2.numpy())
            y_trg = fluid.dygraph.to_variable(y_trg.numpy())
            z_trg = fluid.dygraph.to_variable(z_trg.numpy())
            z_trg2 = fluid.dygraph.to_variable(z_trg2.numpy())

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            #    optims.discriminator.step()
            optims.discriminator.minimize(d_loss)

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.minimize(d_loss)

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.minimize(g_loss)
            optims.mapping_network.minimize(g_loss)
            optims.style_encoder.minimize(g_loss)

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.minimize(g_loss)
            # optims.style_encoder.minimize(g_loss)

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i + 1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # compute FID and LPIPS if necessary  暂时不管
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

        fname = ospj(args.result_dir, 'video_ref.mp4')
        # print('Working on {}...'.format(fname))
        # utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @fluid.dygraph.no_grad
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.stop_gradient = False
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    #  loss_reg = r1_reg(out, x_real)  这个似乎api有问题，先不管

    # with fake images
    with fluid.dygraph.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    # loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    loss = loss_real + loss_fake
    # return loss, Munch(real=loss_real.item(),
    #                    fake=loss_fake.item(),
    #                    reg=loss_reg.item())
    return loss, Munch(real=loss_real.numpy(),
                       fake=loss_fake.numpy())


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
    loss_sty = fluid.layers.reduce_mean(fluid.layers.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()  # x_fake2 = x_fake2.detach()

    loss_ds = fluid.layers.reduce_mean(fluid.layers.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = fluid.layers.reduce_mean(fluid.layers.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc

    return loss, Munch(adv=loss_adv.numpy(),
                       sty=loss_sty.numpy(),
                       ds=loss_ds.numpy(),
                       cyc=loss_cyc.numpy())


def moving_average(source, target, beta=0.999):
    target_model_map = dict(target.named_parameters())

    source_dict = dict(source.named_parameters())
    for param_name in list(source_dict.keys()):
        if 'hpf.conv2' in param_name:
            source_dict.pop(param_name)
    for param_name, source_param in source_dict.items():
        target_param = target_model_map[param_name]
        target_param.set_value(beta * source_param + (1.0 - beta) * target_param)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = fluid.layers.full_like(logits, target)
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, targets)
    loss = fluid.layers.mean(loss)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]

    grad_dout = \
    fluid.dygraph.grad(outputs=fluid.layers.reduce_sum(d_out), inputs=x_in, retain_graph=True, create_graph=True,
                       only_inputs=True)[0]

    grad_dout2 = fluid.layers.pow(grad_dout, factor=2)
    assert (grad_dout2.shape == x_in.shape)
    reg = fluid.layers.reshape(grad_dout2, shape=[batch_size, -1])
    reg = fluid.layers.reduce_sum(reg, dim=1)
    reg = fluid.layers.reduce_mean(reg) * 0.5

    return reg