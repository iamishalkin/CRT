# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.common.losses import *
from collections import OrderedDict


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, data):
        return data.view(data.size(0), -1)


class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=[], pretrain_weight=None):
        super(CNNNet, self).__init__()
        sample_size = data_size['width']
        sample_duration = data_size['depth']
        print (data_size)

        # TODO: Реализуйте архитектуру нейронной сети
        module = nn.Sequential()
        module.add_module('conv_1', nn.Conv2d(3, 16, 5, 1, 2))
        module.add_module('pool_1', nn.MaxPool2d(2, 2))
        module.add_module('conv_2', nn.Conv2d(16, 32, 5, 1, 2))
        module.add_module('pool_1', nn.MaxPool2d(2, 2))
        module.add_module('conv_3', nn.Conv2d(32, 64, 5, 1, 2))
        module.add_module('pool_1', nn.MaxPool2d(2, 2))
        module.add_module('flatten', Flatten())
        module.add_module('linear', nn.Linear(28 * 28 * 64, 100))
        module.add_module('relu', nn.ReLU())
        module.add_module('linear', nn.Linear(100, num_classes))

        self.net = module

    def forward(self, data):
        print(data.shape)
        data = data[:, :, 0, :, :]
        print (data.shape)
        output = self.net(data)
        return output

