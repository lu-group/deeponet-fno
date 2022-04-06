import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io

np.random.seed(1234)

class DataSet:
    def __init__(self, num, bs, modes):
        self.num = num
        self.bs = bs
        self.modes = modes
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std, \
        self.u_basis, self.lam_u = self.load_data()

    def PODbasis(self):
        s = 2295
        print(self.u_basis.shape)

        u_basis_out = np.reshape(self.u_basis.T, (-1, s, 1))
        u_basis_out = self.decoder(u_basis_out)
        u_basis_out = u_basis_out - self.u_mean
        u_basis_out = np.reshape(u_basis_out, (-1, s))
        save_dict = {'u_basis': u_basis_out, 'lam_u': self.lam_u}
        io.savemat('./Output/basis.mat', save_dict)
        return self.u_basis

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x

    def load_data(self):
        num_train = 1900
        num_test = 100

        data = io.loadmat('./Data/Darcy_Triangular')

        s = 101
        r = 2295

        f = data['f_bc']
        u = data['u_field']

        f_train = f[:num_train, :]
        u_train = u[:num_train, :]

        f_test = f[-num_test:, :]
        u_test = u[-num_test:, :]

        xx = data['xx']
        yy = data['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        f_train_mean = np.mean(np.reshape(f_train, (-1, s)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, s)), 0)
        u_train_mean = np.mean(np.reshape(u_train, (-1, r)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, r)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s))
        f_train_std = np.reshape(f_train_std, (-1, 1, s))
        u_train_mean = np.reshape(u_train_mean, (-1, r, 1))
        u_train_std = np.reshape(u_train_std, (-1, r, 1))

        num_res = r
        F_train = np.reshape(f_train, (-1, 1, s))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)       
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)

        F_test = np.reshape(f_test, (-1, 1, s))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)

        U = np.reshape(U_train, (-1, r))
        C_u = 1./(num_train-1)*np.matmul(U.T, U)
        lam_u, phi_u = np.linalg.eigh(C_u)

        lam_u = np.flip(lam_u)
        phi_u = np.fliplr(phi_u)
        phi_u = phi_u*np.sqrt(r)

        u_cumsum = np.cumsum(lam_u)
        u_per = u_cumsum[self.modes-1]/u_cumsum[-1]

        u_basis = phi_u[:, :self.modes]

        print('Kept Energy: %.3f'%(u_per))


        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std, u_basis, lam_u

        
    def minibatch(self):

        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        x_train = self.X

        Xmin = np.array([ 0.,  0.]).reshape((-1, 2))
        Xmax = np.array([ 1.,  1.]).reshape((-1, 2))
        #x_train = np.linspace(-1, 1, self.N).reshape((-1, 1))

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
#        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        batch_id = np.arange(num_test)
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)

        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
