import pickle
import numpy as np
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn


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

    def forward(self, z):
        h = self.shared(z)
        out = []

        for layer in self.unshared.sublayers():
            print(layer)
            out += [layer(h)]
        return out


if __name__ == '__main__':
    place = paddle.fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
    print(place)
    with fluid.dygraph.guard(place):
        net = MappingNetwork()
        z = np.random.random(16).astype('float32')
        z = fluid.dygraph.to_variable(z)
        print(net(z))