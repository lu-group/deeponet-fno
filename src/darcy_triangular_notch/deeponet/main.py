import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from dataset import DataSet
from fnn import FNN
from savedata import SaveData

np.random.seed(1234)
#tf.set_random_seed(1234)

#output dimension of Branch/Trunk
p = 100
num = 101
#branch net
layer_B = [num, 128, 128, p]
#trunk net
layer_T = [2, 128, 128, 128, p]
#resolution
h = num
#batch_size
bs = 100
#size of input for Trunk net
nx = 2295
x_num = nx
epochs = 20000

def main():
    
    data = DataSet(nx, bs)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()
    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]

    f_ph = tf.placeholder(shape=[None, 1, num], dtype=tf.float32) #[bs, f_dim]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])

    fnn_model = FNN()
    # Branch net
    W_B, b_B = fnn_model.hyper_initial(layer_B)
    u_B = fnn_model.fnn_B(W_B, b_B, f_ph)
    u_B = tf.tile(u_B, [1, x_num, 1])   
    #Trunk net
    W_T, b_T = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn_T(W_T, b_T, x, Xmin, Xmax)
    #inner product
    u_nn = u_B*u_T
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

    loss = tf.reduce_mean(tf.square(u_ph - u_pred))
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    saver = tf.train.Saver()
    sess = tf.Session()  
    sess.run(tf.global_variables_initializer())
    
    n = 0
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((epochs+1, 1))
    test_loss = np.zeros((epochs+1, 1))    
    while n <= epochs:
        
        if n < 930:
            lr = 0.001
        elif n < 3000:
            lr = 0.0005
        else:
            lr = 0.0001
            
        x_train, f_train, u_train, _, _ = data.minibatch()
        train_dict={f_ph: f_train, u_ph: u_train, learning_rate: lr}
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)

        if n%1 == 0:
            x_test, f_test, u_test = data.testbatch(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.3e, Test L2 error: %.3f, Time (secs): %.3f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()

        train_loss[n,0] = loss_
        test_loss[n,0] = err
        n += 1

    current_directory = os.getcwd()    
    results_dir = "/Results/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
    
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)      
    
#    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_models_to+'Model')

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
        
    np.savetxt(save_results_to+'/train_loss.txt', train_loss)
    np.savetxt(save_results_to+'/test_loss.txt', test_loss)

    data_save = SaveData()
    num_test = 20
    data_save.save(sess, x_pos, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_test, save_results_to)
    
    ## Plotting the loss history
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
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
    main()
