#import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import sys
np.random.seed(1234)

class DataSet:
    def __init__(self, num, bs):
        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std = self.load_data()

    def func(self, x_train):
        f = np.sin(np.pi*x_train[:, 0:1])*np.sin(np.pi*x_train[:, 1:2])
        u = np.cos(np.pi*x_train[:, 0:1])*np.cos(np.pi*x_train[:, 1:2])
        return f, u

    def samples(self):
        '''
        num_train = 40000
        num_test = 10000
        data = io.loadmat('./Data/Data')
        F = data['F']
        U = data['U']
        '''

        num_train = 1
        x = np.linspace(-1, 1, self.num)
        y = np.linspace(-1, 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        F, U = self.func(x_train)

        Num = self.num*self.num

        F = np.reshape(F, (-1, self.num, self.num, 1))
        U = np.reshape(U, (-1, Num, 1))
        F_train = F[:num_train, :, :]
        U_train = U[:num_train, :, :]
        F_test = F[:num_train, :, :]
        U_test = U[:num_train, :, :]
        return F_train, U_train, F_test, U_test

    def decoder(self, x):
        
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x

    def load_data(self):
        num_train = 1900
        num_test = 100

        data = io.loadmat('./Data/Darcy_Triangular')

        s = 101
        r = 2295
        
        f_train = data['f_bc'][:num_train,:]
        u_train = data['u_field'][:num_train,:]

        f_test = data['f_bc'][num_train:,:]
        u_test = data['u_field'][num_train:,:]
        
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
        
        
        '''
        U_ref = np.reshape(U_test, (U_test.shape[0], U_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std

        
    def minibatch(self):

        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        '''
        x = np.linspace(0., 1, self.num)
        y = np.linspace(0., 1, self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_train = np.hstack((xx, yy))
        '''
        x_train = self.X
#        print(x_train.shape)
#        x_train = np.reshape(x_train, (-1, x_train.shape[0], x_train.shape[1]))
#        print(x_train.shape)
#        sys.exit()
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
        
        '''
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        '''

        '''
        x = np.linspace( 0., 1., self.num)
        y = np.linspace( 0., 1., self.num)
        xx, yy = np.meshgrid(x, y)
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        x_test = np.hstack((xx, yy))
        '''
        x_test = self.X
#
#        batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f_test, u_test
