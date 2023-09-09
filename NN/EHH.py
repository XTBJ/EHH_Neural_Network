import math
import random
from collections import Counter
import os
import sys

import numpy as np
import statsmodels.api as sm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init
from torchvision import datasets
from torchvision import transforms
from data import read_data, load_data
from HH.NN.BReLU import Batch_Norm_Layer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import multiprocessing as mp
import threading as td
import time
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(threshold=np.inf)

# Parameters
epsilon = 0.4

class EHH_Layer(nn.Module):
    def __init__(self, input_size, output_size, intermediate_node=50, quantile=6):
        super(EHH_Layer, self).__init__()
        self.input_size = input_size
        self.quantile = quantile
        self.source_node = input_size*quantile
        self.intermediate_node = intermediate_node
        self.output_node = output_size
        self.batch_layer = Batch_Norm_Layer()
        self.chosen_index = []
        self.w = Parameter(torch.randn(intermediate_node + input_size*quantile, output_size), requires_grad = True)
        self.biases = Parameter(torch.randn(1), requires_grad = True)

    def Maxout_Layer(self, x, num):
        normed, beta, gamma, mean, var = self.batch_layer(x, num)
        output = torch.maximum(torch.tensor(0.0), normed)
        quantile = [-3*var+mean,-0.834*var+mean,-0.248*var+mean,0.248*var+mean,0.834*var+mean]
        sum = []
        sum.append(output)
        for i in range(len(quantile)):
            out = torch.maximum(torch.tensor(0.0), normed-quantile[i]*gamma+beta)
            sum.append(out)
        output = torch.stack(sum, 0)
        return output

    def Minout_Layer(self, x, init_struct=False):
        # define the link between source nodes and intermediate nodes.
        rows = self.source_node
        cols = self.intermediate_node
        n = x.size()[1]
        q = x.size()[2]
        m = self.intermediate_node

        # initialize the EHH structure
        if init_struct:
            out = []
            # Compute the alternative neuron set
            # Only use the information of the source nodes
            self.chosen_index = []
            alter_neurons = {}
            x_after = torch.flatten(x, start_dim=1)

            index_set = set()
            count = 0
            while True:
                n1_index = random.randint(0, n-1)
                n2_index = random.randint(0, n-1)
                while n2_index == n1_index:
                    n2_index = random.randint(0, n-1)
                q1_index = random.randint(0, q-1)
                q2_index = random.randint(0, q-1)
                index = (n1_index, q1_index, n2_index, q2_index)

                while index not in index_set:
                    index_set.add(index)
                    D1 = x[:, n1_index, q1_index]
                    D2 = x[:, n2_index, q2_index]
                    D = torch.stack([D1, D2], 1)
                    v = torch.min(D,1)[0]
                    Jv = []
                    for k in range(n*q):
                        tmp = torch.matmul(v.t(), x_after[:,k])/(torch.norm(x_after[:,k], p=2)*torch.norm(v, p=2)+1e-3)
                        Jv.append(tmp.item())
                    # print("max_Jv:", max(Jv), "     count:", count)
                    if max(Jv) < epsilon:
                        self.chosen_index.append((count,)+index)
                        out.append(v)
                        count += 1
                if count == m:
                    break
        else:
            out = []
            for index in self.chosen_index:
                D1 = x[:, index[1], index[2]]
                D2 = x[:, index[3], index[4]]
                D = torch.stack([D1, D2], 1)
                v = torch.min(D,1)[0]
                out.append(v)
        min_out = torch.stack(out,1)
        return min_out

    def forward(self, x, init_struct):
        max_x = self.Maxout_Layer(x, 1)
        max_x = max_x.permute(1,2,0)
        min_x = self.Minout_Layer(max_x, init_struct)
        x = torch.cat([torch.flatten(max_x, start_dim=1), min_x], 1)
        output = torch.add(torch.matmul(x, self.w), self.biases)

        return output, self.w, max_x, min_x, self.chosen_index

class EHH_Net(nn.Module):
    def __init__(self, input_size, output_size, intermediate_node=20, quantile=6, normalization=True):
        super(EHH_Net, self).__init__()
        self.normalization = normalization
        self.EHH_layer = EHH_Layer(input_size, output_size, intermediate_node, quantile)
        self.adj = self.EHH_layer.chosen_index

    def normalization_layer(self, data):
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        normed = (data-data_min)/(data_max-data_min+1e-5)
        return normed

    def ANOVA_std(self, w, max_x, min_x, test_y):
        scaler = StandardScaler()
        insize = max_x.size()[1]
        q = max_x.size()[2]
        A = []
        B = []
        anova = []
        y_data = test_y.detach().numpy()
        y_data = y_data.reshape(test_y.size()[0], test_y.size()[1])

        for i in range(insize):
            Aw = w[i*q:(i+1)*q]
            Ax = max_x[:,i,:]
            A_tmp = torch.matmul(Ax, Aw).detach().numpy()
            A.append(A_tmp)

        for j in range(min_x.size()[1]):
            Bw = w[insize*q+j].unsqueeze(0)
            Bx = min_x[:,j].unsqueeze(1)
            B_tmp = np.array(torch.matmul(Bx, Bw).cpu().detach().numpy())
            B.append(B_tmp)

        A = np.array(A)
        B = np.array(B)

        for dim in range(test_y.size()[1]):
            y = y_data[:, dim]
            y = y[:,np.newaxis]
            x = np.concatenate((A[:,:,dim],B[:,:,dim]), axis=0).swapaxes(0,1)

            X_scaled = scaler.fit_transform(x)
            Y_scaled = scaler.fit_transform(y)

            reg = LinearRegression()
            reg.fit(X_scaled, Y_scaled)

            coef_scaled = np.abs(reg.coef_)[0]

            # max-min
            std_max = max(coef_scaled)
            std_min = min(coef_scaled)
            std_att = [(x-std_min)/(std_max-std_min) for x in coef_scaled]
            anova.append(std_att)

        anova = np.squeeze(anova)
        return anova

    def ANOVA(self, w, max_x, min_x, test_y):
        A = []
        B = []
        anova = []
        insize = max_x.size()[1]
        q = max_x.size()[2]
        y = np.array(test_y.cpu().detach().numpy()).reshape(test_y.size()[0], test_y.size()[1])
        for i in range(insize):
            Aw = w[i*q:(i+1)*q]
            Ax = max_x[:,i,:]
            A_tmp = np.array(torch.matmul(Ax, Aw).cpu().detach().numpy())
            A.append(A_tmp)
            formula1 = 'y~' + '+'.join('A{}'.format(k) for k in range(0, i+1))

        for j in range(min_x.size()[1]):
            Bw = w[insize*q+j].unsqueeze(0)
            Bx = min_x[:,j].unsqueeze(1)
            B_tmp = np.array(torch.matmul(Bx, Bw).cpu().detach().numpy())
            B.append(B_tmp)
            formula2 = '+'.join('B{}'.format(k) for k in range(0, j+1))

        A = np.array(A)
        B = np.array(B)

        for dim in range(test_y.size()[1]):
            d = {}
            d['y'] = y[:, dim, None]
            for i in range(insize):
                d['A%d'%i] = A[i, :, dim, None]
            for j in range(min_x.size()[1]):
                d['B%d'%j] = B[j,:, dim, None]
            
            formula = formula1+'+'+formula2
            model = sm.formula.ols(formula, d).fit()
            F = sm.stats.anova_lm(model)['F'].tolist()
            F = F[:len(F)-1]

            # max-min
            F_max = max(F)
            F_min = min(F)
            F_att = [(x-F_min)/(F_max-F_min) for x in F]
            anova.append(F_att)

        return anova

    def forward(self,x,init_struct):
        if self.normalization:
            normed = self.normalization_layer(x)
        else:
            normed = x
        output, w, max_x, min_x, adj = self.EHH_layer(normed, init_struct)
        return output, w, max_x, min_x, adj
