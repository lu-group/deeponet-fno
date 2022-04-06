import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import scipy
from fnn import FNN

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, f_ph, u_ph, u_pred, data, num_test, h):

        test_id, x_test, f_test, u_test = data.testbatch(num_test)
        test_dict = {f_ph: f_test, u_ph: u_test}

        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)
        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error: %.3f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err', err, fmt='%e')

        save_dict = {'u_pred': u_pred_, 'u_ref': U_ref, 'f_test': f_test, 'test_id': test_id}
        io.savemat('./Output/pred.mat', save_dict)
        
        scipy.io.savemat('./Output/darcy_triangular_test_DeepONetPOD.mat', 
                     mdict={'x_test': f_test,
                            'y_test': u_test, 
                            'y_pred': u_pred_})

