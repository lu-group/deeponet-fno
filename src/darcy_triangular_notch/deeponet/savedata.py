import tensorflow.compat.v1 as tf
import numpy as np
import sys
from fnn import FNN
import os
import matplotlib.pyplot as plt
import scipy
plot_folder = './Plot/'    
class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_test, save_results_to):
        
        x_test, f_test, u_test = data.testbatch(num_test)
        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn_T(W_T, b_T, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, u_ph: u_test}
        u_B = fnn_model.fnn_B(W_B, b_B, f_ph)
        u_B = tf.tile(u_B, [1, x_test.shape[0], 1])
        u_nn = u_B*u_T
        '''
        xs = 2*(x - Xmin)/(Xmax - Xmin) - 1
        u_pred = (xs[:, :, 0:1] - 1)*(xs[:, :, 0:1] + 1)*(xs[:, :, 1:2] - 1)*(xs[:, :, 1:2] + 1)*u_pred
        '''
        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)   

        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))        
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error: %.3f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_results_to+'/err.txt', err, fmt='%e')
        
        scipy.io.savemat(save_results_to+'darcy_triangular_test_DeepONet.mat', 
                     mdict={'x_test': f_test,
                            'y_test': u_test, 
                            'y_pred': u_pred_})