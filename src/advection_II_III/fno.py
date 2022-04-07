from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities3 import count_params, LpLoss, UnitGaussianNormalizer


class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
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

    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x, y), x)
        # Add t channel
        self.fc0 = nn.Linear(3, self.width) # input channel is 2: (a(x, y), x, t)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
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
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def main():
    batch_size = 20
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    # Using more modes or width doesn't help.
    modes1 = 16  # At most nt
    modes2 = 16  # At most nx / 2 + 1
    width = 64  # Max = 190 in NVIDIA GeForce RTX 2080 Ti, otherwise OOM

    ################################################################
    # load data and data normalization
    ################################################################
    nt = 40
    nx = 40
    ntrain = 1000
    ntest = 1000

    data = np.load('train_IC1.npz')
    x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
    u0_train = u_train[:, 0, :]  # N x nx
    x_normalizer = UnitGaussianNormalizer(torch.from_numpy(u0_train))
    u0_train = x_normalizer.encode(torch.from_numpy(u0_train)).numpy()
    y_normalizer = UnitGaussianNormalizer(torch.from_numpy(u_train))
    u_train = y_normalizer.encode(torch.from_numpy(u_train)).numpy()

    x = np.repeat(x[0:1, :], ntrain, axis=0)  # N x nx
    x_train = np.concatenate((u0_train[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
    x_train = np.repeat(x_train[:, None, :, :], nt, axis=1)  # N x nt x nx x 2
    # Add t channel
    t = np.repeat(t.reshape(1, nt, nx), ntrain, axis=0)  # N x nt x nx
    x_train = np.concatenate((x_train, t[:, :, :, None]), axis=-1)  # N x nt x nx x 3
    x_train = torch.from_numpy(x_train)
    u_train = torch.from_numpy(u_train)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, u_train), batch_size=batch_size, shuffle=True)

    data = np.load('test_IC1.npz')
    x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
    u0_test = u_test[:, 0, :]  # N x nx
    u0_test = x_normalizer.encode(torch.from_numpy(u0_test)).numpy()
    x = np.repeat(x[0:1, :], ntest, axis=0)  # N x nx
    x_test = np.concatenate((u0_test[:, :, None], x[:, :, None]), axis=-1)  # N x nx x 2
    x_test = np.repeat(x_test[:, None, :, :], nt, axis=1)  # N x nt x nx x 2
    # Add t channel
    t = np.repeat(t.reshape(1, nt, nx), ntest, axis=0)  # N x nt x nx
    x_test = np.concatenate((x_test, t[:, :, :, None]), axis=-1)  # N x nt x nx x 3
    x_test = torch.from_numpy(x_test)
    u_test = torch.from_numpy(u_test)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, u_test), batch_size=batch_size, shuffle=False)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes1, modes2, width).cuda()
    print(count_params(model))

    # Remove weight_decay doesn't help.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, nt, nx)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            # loss.backward()
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()
    
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x).reshape(batch_size, nt, nx)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        train_mse /= len(train_loader)
        train_l2/= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                  % ( ep, t2-t1, train_mse, train_l2, test_l2), flush=True)

    elapsed = default_timer() - start_time
    print('Training time: %.3f'%(elapsed))

    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.cuda()
        out = model(x).reshape(batch_size, nt, nx)
        out = y_normalizer.decode(out).cpu().numpy()
        y = y.numpy()
    np.savetxt("y_pred_fno.dat", out[0])
    np.savetxt("y_true_fno.dat", y.reshape(batch_size, nt, nx)[0])
    np.savetxt("y_error_fno.dat", out[0] - y.reshape(batch_size, nt, nx)[0])


if __name__ == "__main__":
    main()
