# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import os
import paddle
import paddle.fluid as fluid


class CheckpointIO(object):
    def __init__(self, fname_template):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template

    def save(self, step, module, module_class):
        assert module_class in ['para', 'opti']
        fname = self.fname_template.format(step)
        print(f'Saving checkpoint of {module_class} into {fname}...')
        state_dict = module.state_dict()
        fluid.dygraph.save_dygraph(state_dict, fname)

    def load(self, step, module=None, module_class='all'):
        assert module_class in ['all', 'para', 'opti']
        fname = self.fname_template.format(step)
        # assert os.path.exists(fname + '.pdparams'), fname + ' does not exist!'
        print(f'Loading checkpoint of {module_class} from {fname}...')
        para_state_dict, opti_state_dict = fluid.dygraph.load_dygraph(fname)
        if module is None:
            return para_state_dict, opti_state_dict
        if module_class == 'para':
            return module.load_dict(para_state_dict)
        if module_class == 'opti':
            return module.load_dict(opti_state_dict)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    place = paddle.fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
    checkpoint = CheckpointIO('./{:06d}_train')
    with fluid.dygraph.guard(place):
        emb = fluid.dygraph.Embedding([10, 10])
        checkpoint.save(1, emb, 'para')
        state_dict = emb.state_dict()
        fluid.save_dygraph(state_dict, "paddle_dy")

        adam = fluid.optimizer.Adam(learning_rate=fluid.layers.noam_decay(100, 10000),
                                    parameter_list=emb.parameters())
        checkpoint.save(1, adam, 'opti')
        state_dict = adam.state_dict()
        fluid.save_dygraph(state_dict, "paddle_dy")

        para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
        para_state_dict1, opti_state_dict1 = checkpoint.load(1)

        emb1 = fluid.dygraph.Embedding([10, 10])
        emb1.load_dict(para_state_dict1)

        assert (para_state_dict['weight'] == para_state_dict1['weight']).all()
