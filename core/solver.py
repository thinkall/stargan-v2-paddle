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

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_reader import InputFetcher


class Solver(fluid.dygraph.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args