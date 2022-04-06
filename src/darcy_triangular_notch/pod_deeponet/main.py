import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset import DataSet
from fnn import FNN
from conv import CNN
from savedata import SaveData

# Remove existing results
if os.path.exists('Output'):
    os.system('rm -r Output/')

os.makedirs('Output/')    

np.random.seed(1234)

#output dimension of Branch/Trunk
p = 100

modes = 20
s = 101
s_in = 1600
#fnn in CNN
layer_linear = [s, s_in]
layer_B = [40, modes]
#trunk net
layer_T = [2, 128, 128, 128, p]

#resolution
h = 40
w = 40

#parameters in CNN
n_channels = 1
n_out_channels = 16
filter_size_1 = 5
filter_size_2 = 3
stride = 1

#filter size for each convolutional layer
num_filters_1 = 16
num_filters_2 = 16
num_filters_3 = 64

#batch_size
bs = 50

#size of input for Trunk net
nx = 2295
x_num = nx

def main():
    data = DataSet(nx, bs, modes)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()
    u_basis = data.PODbasis()  

    f_ph = tf.placeholder(shape=[None, 1, s], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    
    #Branch net
    conv_model = CNN()
    fnn_model = FNN()

    W_l, b_l = fnn_model.hyper_initial(layer_linear)
    conv_linear = fnn_model.fnn_B(W_l, b_l, f_ph)
    conv_linear = tf.reshape(conv_linear, [-1, s_in])
    conv_linear = tf.reshape(conv_linear, [-1, h, w])
    conv_linear = tf.tile(conv_linear[:, :, :, None], [1, 1, 1, 1])

    conv_1 = conv_model.conv_layer(conv_linear, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2) 
    conv_2 = conv_model.conv_layer(pool_1, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    layer_flat = conv_model.flatten_layer(pool_2)
    fnn_layer_1 = conv_model.fnn_layer(layer_flat, layer_B[0], actn=tf.tanh, use_actn=True)
    out_B = conv_model.fnn_layer(fnn_layer_1, layer_B[1], actn=tf.tanh, use_actn=False) #[bs, p]
   
    #POD basis
    u_basis = tf.constant(u_basis, dtype=tf.float32)
    
    #prediction
    u_pred = tf.einsum('bi,ni->bn', out_B, u_basis)
    u_pred = tf.tile(u_pred[:, :, None], [1, 1, 1])

    loss = tf.reduce_mean(tf.square(u_ph - u_pred))
    train = tf.train.AdamOptimizer(learning_rate=1.0e-3).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n = 0
    nmax = 20000
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:

        x_train, f_train, u_train, _, _ = data.minibatch()
        train_dict={f_ph: f_train, u_ph: u_train}
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)

        if n%100 == 0:
            test_id, x_test, f_test, u_test = data.testbatch(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.3e, Test L2 error: %.3e, Time (secs): %.3f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()

        n += 1

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))

    data_save = SaveData()
    num_test = 100
    data_save.save(sess, f_ph, u_ph, u_pred, data, num_test, h)

if __name__ == "__main__":
    main()
