from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import count_params, LpLoss


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def main():
    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    modes = 16
    width = 32

    ################################################################
    # load data
    ################################################################
    nt = 40
    nx = 40
    ntrain = 1000
    ntest = 1000

    data = np.load('train_IC2.npz')
    x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
    u0_train = u_train[:, 0, :]  # N x nx
    x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
    x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
    u_train = u_train.transpose(0, 2, 1)  # N x nx x nt
    x_train = torch.from_numpy(x_train)
    u_train = torch.from_numpy(u_train)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)

    data = np.load('test_IC2.npz')
    x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
    u0_test = u_test[:, 0, :]  # N x nx
    x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
    x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
    u_test = u_test.transpose(0, 2, 1)  # N x nx x nt
    x_test = torch.from_numpy(x_test)
    u_test = torch.from_numpy(u_test)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    device = torch.device('cuda')
    model = FNO1d(modes, width).cuda()
    print(count_params(model))

    # Remove weight_decay doesn't help.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_l2 = 0
        train_mse = 0
        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            optimizer.zero_grad()

            pred = xx[:, :, :1]  # t = 0
            for t in range(1, nt, 1):
                im = model(xx)
                pred = torch.cat((pred, im), -1)
                xx = torch.cat((im, xx[:, :, 1:]), dim=-1)  # cat x

            mse = F.mse_loss(pred.view(batch_size, -1), yy.view(batch_size, -1), reduction='mean')
            mse.backward()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2_full.item()

        scheduler.step()
        model.eval()
        test_l2 = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred = xx[:, :, 0:1]
                for t in range(1, nt, 1):
                    im = model(xx)
                    pred = torch.cat((pred, im), -1)
                    xx = torch.cat((im, xx[:, :, 1:]), dim=-1)
                test_l2 += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
        t2 = default_timer()
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                % (ep, t2-t1, train_mse, train_l2, test_l2), flush=True)

    elapsed = default_timer() - start_time
    print('Training time: %.3f'%(elapsed))

    with torch.no_grad():
        xx, yy = next(iter(test_loader))
        xx = xx.cuda()
        pred = xx[:, :, 0:1]
        for t in range(1, nt, 1):
            im = model(xx)
            pred = torch.cat((pred, im), -1)
            xx = torch.cat((im, xx[:, :, 1:]), dim=-1)
        out = pred.permute(0, 2, 1).cpu().numpy()
        y = yy.permute(0, 2, 1).numpy()
    np.savetxt("y_pred_fnotime.dat", out[0])
    np.savetxt("y_true_fnotime.dat", y[0])
    np.savetxt("y_error_fnotime.dat", out[0] - y[0])


if __name__ == "__main__":
    main()
