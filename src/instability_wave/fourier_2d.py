"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import scipy


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def interpolate(DATA_PATH, data):
    # Interpolate the input (t, y) resolution from (20, 47) to (111, 47)
    rt = np.load(DATA_PATH + "rt.npy")  # (20,) uniform
    ry = np.load(DATA_PATH + "ry.npy")  # (47,) nonuniform
    # rx = np.load(DATA_PATH + "rx.npy")  # (111,) uniform
    # Normalize
    rt = (rt - np.min(rt)) / (np.max(rt) - np.min(rt))
    ry = (ry - np.min(ry)) / (np.max(ry) - np.min(ry))

    tt, yy = np.meshgrid(rt, ry, indexing='ij')  # (20, 47)

    rt_min, rt_max = np.min(rt), np.max(rt)
    rt_new = np.linspace(rt_min, rt_max, num=111)
    tt_new, yy_new = np.meshgrid(rt_new, ry, indexing='ij')  # (111, 47)
    data_new = []
    points = np.hstack((tt.reshape(-1, 1), yy.reshape(-1, 1)))
    xi = np.hstack((tt_new.reshape(-1, 1), yy_new.reshape(-1, 1)))
    for x in data:
        data_new.append(griddata(points, np.ravel(x), xi))
    data_new = np.array(data_new).reshape(-1, 111, 47)
    # plt.figure()
    # plt.imshow(data[0])
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(data_new[0])
    # plt.colorbar()
    # plt.show()

    grid = np.concatenate((tt_new[:, :, None], yy_new[:, :, None]), axis=-1)  # (111, 47, 2)
    return data_new, grid


def weighted_mse_loss(input, target):
    A = torch.max(target, 1)[0]
    loss = torch.mean((input - target) ** 2, 1)
    return torch.mean(loss / A)


def main():
    ################################################################
    # configs
    ################################################################
    DATA_PATH = 'instability_wave/'
    
    batch_size = 100
    learning_rate = 0.001

    epochs = 700
    step_size = 100
    gamma = 0.5
    
    modes = 16
    width = 64

    ################################################################
    # load data and data normalization
    ################################################################
    x_train = np.load(DATA_PATH + "X_train.npy")  # (40800, 20, 47), (t, y)
    y_train = np.load(DATA_PATH + "Y_train.npy")  # (40800, 111, 47), (x, y) 
    # w_train = np.load(DATA_PATH + "W_train.npy")[:1000]  # (40800,)
    x_valid = np.load(DATA_PATH + "X_valid.npy")  # (10000, 20, 47)
    y_valid = np.load(DATA_PATH + "Y_valid.npy")  # (10000, 111, 47)
    # w_valid = np.load(DATA_PATH + "W_valid.npy")  # (10000,)

    print("Interpolating train...", flush=True)
    x_train, grid = interpolate(DATA_PATH, x_train)  # (40800, 111, 47), (111, 47, 2)
    print("Interpolate valid...", flush=True)
    x_valid, grid = interpolate(DATA_PATH, x_valid)  # (10000, 111, 47), (111, 47, 2)
    grid = grid.reshape(1, 111, 47, 2)

    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    # w_train = torch.from_numpy(w_train.astype(np.float32))
    x_valid = torch.from_numpy(x_valid.astype(np.float32))
    y_valid = torch.from_numpy(y_valid.astype(np.float32))
    # w_valid = torch.from_numpy(w_valid.astype(np.float32))
    grid = torch.from_numpy(grid.astype(np.float32))

    # x_normalizer = UnitGaussianNormalizer(x_train)
    # x_train = x_normalizer.encode(x_train)
    # x_valid = x_normalizer.encode(x_valid)
    y_normalizer = UnitGaussianNormalizer(y_train)

    x_train = torch.cat([x_train.reshape(-1, 111, 47, 1), grid.repeat(len(x_train), 1, 1, 1)], dim=3)  # (40800, 111, 47, 3)
    x_valid = torch.cat([x_valid.reshape(-1, 111, 47, 1), grid.repeat(len(x_valid), 1, 1, 1)], dim=3)  # (10000, 111, 47, 3)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, 111, 47)
            out = y_normalizer.decode(out)

            loss = weighted_mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x).reshape(batch_size, 111, 47)
                out = y_normalizer.decode(out)
                loss = weighted_mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
                valid_loss += loss.item()
        
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        t2 = default_timer()
        print("Epoch: %d, time: %.3f, Train loss: %.4f, Valid loss: %.4f" 
                  % ( ep, t2-t1, train_loss, valid_loss), flush=True)

    ################################################################
    # Evaluate tests with different noise amplitudes
    ################################################################
    model.cpu()
    y_normalizer.cpu()

    output = open("fno_errs_noise.dat", "w")
    amplitudes = [0.0, 0.001, 0.01, 0.1, 0.2]
    Nm = 300
    Na = len(amplitudes)
    acc_err = np.zeros((Nm, Na))
    for ii in range(Nm):
        ev = np.load(DATA_PATH + f"tests/eig_vec_{ii:02}.npy")
        pd = np.load(DATA_PATH + f"tests/prof_data_{ii:02}.npy")

        for ia, amp in enumerate(amplitudes):
            if ia == 0:
                eta0 = np.random.standard_normal(np.shape(ev))
                eta = np.max(ev) * eta0

            noise = amp * eta
            evn = ev + noise

            evn = np.expand_dims(evn, axis=0)
            x, grid = interpolate(DATA_PATH, evn)
            x = torch.from_numpy(x.astype(np.float32))
            grid = torch.from_numpy(grid.astype(np.float32))
            # x = x_normalizer.encode(x)
            x = torch.cat([x.reshape(1, 111, 47, 1), grid.reshape(1, 111, 47, 2)], dim=3)
            out = model(x).reshape(1, 111, 47)
            po = y_normalizer.decode(out).detach().numpy()

            err = np.mean((po - pd) ** 2)
            rel = np.sqrt(err / np.mean(pd ** 2))
            acc_err[ii, ia] = rel

    errs = acc_err.mean(axis=0)
    bars = acc_err.std(axis=0)
    for ia in range(Na):
        print(errs[ia], bars[ia], file=output)
    output.close()


if __name__ == "__main__":
    main()
