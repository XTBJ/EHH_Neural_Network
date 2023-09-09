import math
import random
from collections import Counter
import pandas as pd
import os
import sys
sys.path.append("/home/lzy/demo/")
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

import multiprocessing as mp
import threading as td
import time
from matplotlib import pyplot as plt
from NN.EHH import EHH_Net

torch.set_printoptions(threshold=np.inf)

# UCI/winequanlity-dataset
def get_data():
    data = load_data.read_data_sets(torch.float32,"data//UCI//winequality-white1.csv")
    return data
data = get_data()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True

# Parameters
learning_rate = 1e-3
training_epochs = 50
batch_size = 128
random_epochs = (data.train.images.shape[0]//batch_size)+1
display_step = 1
epsilon = 0.4

class ReLU_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReLU_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        output = x
        return output

# ReLU model
ReLU_model = ReLU_Net(data.train.images.shape[1], 36, data.train.labels.shape[1]).to(device)
ReLU_optimizer = torch.optim.Adam(ReLU_model.parameters(), lr=learning_rate)
ReLU_model.train()

# EHH model
EHH_model = EHH_Net(data.train.images.shape[1], data.train.labels.shape[1], intermediate_node=20, normalization=True).to(device)
EHH_optimizer = torch.optim.Adam(EHH_model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()
# criterion = nn.MSELoss()
EHH_model.train()

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    # plt.title('train loss')
    plt.title('error')
    plt.show()

def Run_ReLU():
    train_loss = []
    train_error = []
    for ep in range(training_epochs):
        ReLU_optimizer.zero_grad()
        for num in range(random_epochs):
            batch_x, batch_y = data.train.next_batch(batch_size)
            train_x = Variable(torch.tensor(batch_x, dtype=torch.float32).to(device))
            train_y = Variable(torch.tensor(batch_y, dtype=torch.float32).to(device))

            pred = ReLU_model(train_x)
            lasso_loss = criterion(pred, train_y)
            lasso_loss.backward()
            ReLU_optimizer.step()
            train_loss.append(lasso_loss.item())

        ReLU_model.eval()
        
        # wine-quantity
        batch_index = np.random.randint(0,len(data.test.images)-1, size=batch_size)
        test_x = Variable(torch.tensor(data.test.images[batch_index], dtype=torch.float32).to(device))
        test_y = Variable(torch.tensor(data.test.labels[batch_index], dtype=torch.float32).to(device))

        pred = ReLU_model(test_x)
        
        # wine-quantity        
        MSELoss = torch.nn.MSELoss()
        error = MSELoss(pred, test_y)
        if ep>training_epochs-20:
            train_error.append(error.item())
        print("ep:", ep, "  error:", error)

    plot_curve(train_error)

        
def Run_EHH():
    out, w, max1, min1, adj = EHH_model(torch.tensor(data.train.images, dtype=torch.float32).to(device), init_struct=True)
    train_loss = []
    train_error = []
    for ep in range(training_epochs):
        # print("-------ep:%d--------" %ep)
        EHH_optimizer.zero_grad()
        for num in range(random_epochs):
            # print("-------num:%d--------" %num)
            batch_x, batch_y = data.train.next_batch(batch_size)
            train_x = Variable(torch.tensor(batch_x, dtype=torch.float32).to(device))
            train_y = Variable(torch.tensor(batch_y, dtype=torch.float32).to(device))

            pred, w, max2, min2, adj = EHH_model(train_x, init_struct=False)
            GCV = torch.sum(torch.square(train_y.detach()-pred.detach()))/(torch.tensor(batch_size, dtype=torch.float)*torch.square(torch.tensor(1-(len(w)+1)/batch_size, dtype=torch.float)))

            lamb = GCV*torch.sqrt(2*torch.log(torch.tensor(len(w))))

            lasso_loss = criterion(pred, train_y) + lamb*torch.norm(w, p=1)
            lasso_loss.backward()
            EHH_optimizer.step()
            train_loss.append(lasso_loss.item())

        EHH_model.eval()

        # wine-quantity
        batch_index = np.random.randint(0,len(data.test.images)-1, size=batch_size)
        test_x = Variable(torch.tensor(data.test.images[batch_index], dtype=torch.float32).to(device))
        test_y = Variable(torch.tensor(data.test.labels[batch_index], dtype=torch.float32).to(device))

        pred, w, max_x, min_x, adj = EHH_model(test_x, init_struct=False)
        
        if ep == training_epochs-1:
            anova = EHH_model.ANOVA(w, max_x, min_x, test_y)
            print(anova)

        # wine-quantity
        MSELoss = torch.nn.MSELoss()
        error = MSELoss(pred, test_y)
        if ep>training_epochs-20:
            train_error.append(error.item())
        print("ep:", ep, "  error:", error)
        
    plot_curve(train_error)

        
def main():
    print("---------%s---------" %"Results on ReLU")
    Run_ReLU()
    print("---------%s---------" %"Results on EHH")
    Run_EHH()


if __name__ =='__main__':
    main()
