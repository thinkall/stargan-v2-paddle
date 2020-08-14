# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import os
import paddle
import paddle.fluid as fluid


place = paddle.fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()


class CheckpointIO(object):
    def __init__(self, fname_template, train_model):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.train_model = train_model

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        with fluid.dygraph.guard(place):
            state_dict = self.train_model.state_dict()
            fluid.dygraph.save_dygraph(state_dict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        with fluid.dygraph.guard(place):
            model, _ = fluid.dygraph.load_dygraph(fname)
            self.train_model.load_dict(model)
