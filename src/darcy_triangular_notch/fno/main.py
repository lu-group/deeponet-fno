"""
@author: Somdatta Goswami
This file is the Fourier Neural Operator for 2D Darcy Problem that has a notch in the domain.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy
from plotting import *
import time
import sys
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")

np.random.seed(0)

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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype = torch.cfloat))

    #Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = torch.einsum("bixy,ioxy->boxy",a,b)
        return op

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        #Return to physical space
        x = torch.fft.irfft2(out_ft,s=(x.size(-2),x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()

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

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes,  width)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

def FNO_main(save_index):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """
    ################################################################
    # configs
    ################################################################
    PATH = 'Data/Darcy_Triangular_FNO.mat'
 
    ntrain = 1900
    ntest = 100
    
    batch_size = 10
    learning_rate = 0.001
    
    epochs = 800
    step_size = 100
    gamma = 0.5
    
    r = 2
    h = int(((100 - 1)/r) + 1)
    s = h
    
    modes = 8
    width = 32
    
    ################################################################
    # load data and data normalization
    ################################################################
    reader = MatReader(PATH)
    x_train = reader.read_field('boundCoeff')[:ntrain,::r,::r][:,:s,:s]
    y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
    grid_x_train = reader.read_field('coord_x')[:ntrain,::r,::r][:,:s,:s]
    grid_y_train = reader.read_field('coord_y')[:ntrain,::r,::r][:,:s,:s]
    
    reader.load_file(PATH)
    x_test = reader.read_field('boundCoeff')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
    y_test = reader.read_field('sol')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
    grid_x_test = reader.read_field('coord_x')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
    grid_y_test = reader.read_field('coord_y')[ntrain:ntrain+ntest,::r,::r][:,:s,:s]
    
    grid_x_train = grid_x_train.reshape(ntrain, s, s, 1)
    grid_y_train = grid_y_train.reshape(ntrain, s, s, 1)
    x_train = x_train.reshape(ntrain, s, s, 1)
    x_train = torch.cat([x_train, grid_x_train, grid_y_train], dim = -1)
    
    grid_x_test = grid_x_test.reshape(ntest, s, s, 1)
    grid_y_test = grid_y_test.reshape(ntest, s, s, 1)
    x_test = x_test.reshape(ntest, s, s, 1)
    x_test = torch.cat([x_test, grid_x_test, grid_y_test], dim = -1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    
    train_loader_L2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=ntrain, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    ################################################################
    # training and evaluation
    ################################################################
    model = Net2d(modes, width).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start_time = time.time()
    myloss = LpLoss(size_average=False)
#    y_normalizer.cuda()
    
    train_loss = np.zeros((epochs, 1))
    test_loss = np.zeros((epochs, 1))
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            loss.backward()
            optimizer.step()
            train_mse += loss.item()        
        
        scheduler.step()
        model.eval() 
        
        train_L2 = 0
        with torch.no_grad():
            for x, y in train_loader_L2:
                x, y = x.cuda(), y.cuda() 
                out = model(x)
                l2 = myloss(out.view(ntrain, -1), y.view(ntrain, -1)) 
                train_L2 += l2.item() 
                
        test_L2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                test_L2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
                
        train_mse /= len(train_loader)
        train_L2 /= ntrain
        test_L2 /= ntest
        train_loss[ep,0] = train_L2
        test_loss[ep,0] = test_L2
        
        t2 = default_timer()

        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" % ( ep, t2-t1, train_mse, train_L2, test_L2))
          
    elapsed = time.time() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")
    
    # ====================================
    # saving settings
    # ====================================
    current_directory = os.getcwd()
    case = "Case_"
    folder_index = str(save_index)
    
    results_dir = "/" + case + folder_index +"/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
    
    np.savetxt(save_results_to+'/train_loss.txt', train_loss)
    np.savetxt(save_results_to+'/test_loss.txt', test_loss)
        
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)
        
    torch.save(model, save_models_to+'Darcy')

    ################################################################
    # testing
    ################################################################
    dump_test =  save_results_to+'/Predictions/'
    os.makedirs(dump_test, exist_ok=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)    
    pred = torch.zeros(ntest,s,s)
    index = 0
    test_l2 = 0
    t1 = default_timer()
    dataSegment = "Test"
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            pred[index,:,:] = out
            perm_print = x[0,:,:,0].cpu().numpy()
            disp_pred = out.cpu().numpy()
            disp_true = y[0,:,:].cpu().numpy()
            plotField(perm_print, disp_pred, disp_true, index, dump_test, dataSegment) 
            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            index = index + 1
    
    test_l2 = test_l2/index
    t2 = default_timer()
    testing_time = t2-t1

    scipy.io.savemat(save_results_to+'darcy_test.mat', 
                     mdict={'x_test': x_test.numpy(),
                            'y_test': y_test.numpy(), 
                            'y_pred': pred.cpu().numpy(),
                            'testing_time': testing_time})  
    
    
    print("\n=============================")
    print('Testing error: %.3e'%(test_l2))
    print("=============================\n")
    
    # Plotting the loss history
    num_epoch = epochs
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_his.png')

if __name__ == "__main__":
    
    save_index = 1
    FNO_main(save_index)
