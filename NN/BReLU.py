'''
@ Summary: BReLU Neural Network, Pytorch Version
@ Author: XTBJ
'''

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from NN import load_data

# hyper-parameters
LR = 0.01
scale = 0.02

class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0-self.decay)*x + self.decay*self.shadow[name]
        self.shadow[name] = new_average.clone()

class Batch_Norm_Layer(nn.Module):
    def __init__(self):
        super(Batch_Norm_Layer, self).__init__()
        self.beta = Parameter(torch.zeros([1]))
        self.gamma = Parameter(torch.ones([1]))
        self.ema = EMA(decay=0.5)
        # self.bn = nn.BatchNorm1d(num_feature, eps=1e-3, momentum=0.5)

    def forward(self, x, num):
        axis = list(range(len(x.shape)-1))
        batch_mean = torch.mean(x, [0])
        batch_var = torch.var(x, [0], unbiased=False)
        data = {"mean"+str(num): batch_mean, "var"+str(num): batch_var}

        for name in data:
            self.ema.register(name, data[name])

        for name in data:
            self.ema.update(name, data[name])
            mean = self.ema.get("mean"+str(num))
            var = self.ema.get("var"+str(num))
            # print(name,":",ema.get(name))
        normed = self.gamma*(x-mean)/torch.sqrt(var + torch.tensor(1e-3)) + self.beta
        return normed, self.beta, self.gamma, mean, var

class BReLU_Layer(nn.Module):
    def __init__(self):
        super(BReLU_Layer, self).__init__()
        self.batch_layer = Batch_Norm_Layer()

    def forward(self, x, num):
        normed, beta, gamma, mean, var = self.batch_layer(x, num)
        output = torch.maximum(torch.tensor(0.0), normed)
        quantile = [-3*var+mean,-0.834*var+mean,-0.248*var+mean,0.248*var+mean,0.834*var+mean]
        for i in range(len(quantile)):
            out = torch.maximum(torch.tensor(0.0), normed-quantile[i]*gamma+beta)
            output = torch.cat([output, out], 1)
        return output

class BReLU_Net(nn.Module):
    def __init__(self, in_size, hidden_size1, hidden_size2, out_size, q):
        super(BReLU_Net, self).__init__()
        self.w1 = Parameter(torch.randn(in_size, hidden_size1), requires_grad = True)
        self.w2 = Parameter(torch.randn(hidden_size1*(q+1), hidden_size2), requires_grad = True)
        self.w3 = Parameter(torch.randn(hidden_size2*(q+1), out_size), requires_grad = True)
        self.batch_layer1 = BReLU_Layer()
        self.batch_layer2 = BReLU_Layer()
        self.biases = Parameter(torch.zeros([1]))

    def forward(self, x):
        # x1 = torch.matmul(self.w1,x)
        x1 = torch.matmul(x, self.w1)
        out1 = self.batch_layer1(x1, 1)
        output1 = [out1]
        output1 = torch.concat(output1,1)
        in_size1 = output1.shape[1]

        x2 = torch.matmul(output1, self.w2)
        out2 = self.batch_layer2(x2, 2)
        output2 = [out2]
        output2 = torch.concat(output2,1)
        in_size2 = output2.shape[1]

        pre_y = torch.add(torch.matmul(output2, self.w3), self.biases)
        return pre_y, self.w1, self.w2, self.w3
