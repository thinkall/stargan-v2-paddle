import os

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Conv2DTranspose, InstanceNorm, PRelu


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        os.makedirs(fname, exist_ok=True)
        for name, module in self.module_dict.items():

            if 'optims' in fname:
                break
            outdict[name] = module.state_dict()
            fluid.dygraph.save_dygraph(outdict[name], fname + '/' + name)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        os.makedirs(fname, exist_ok=True)
        for name in os.listdir(fname):
            module_dict, _ = fluid.load_dygraph(fname + '/' + name)
            self.module_dict[name.split('.', 1)[0]].load_dict(module_dict)
    #  module_dict,_=fluid.load_dygraph(fname)
    #  model.load_dict(model_dict)#加载模型参数
    # model.eval()#评估模式
    # for name, module in self.module_dict.items():
    #     module.load_dict(module_dict[name])
