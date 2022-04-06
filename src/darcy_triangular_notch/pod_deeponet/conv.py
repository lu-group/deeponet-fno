import tensorflow.compat.v1 as tf
import numpy as np

class CNN:
    def __init__(self):
        pass

    def hyper_initial(self, shape_w, shape_b):
        std = 0.01
        weight = tf.Variable(tf.random_normal(shape=shape_w, stddev=std), dtype=tf.float32)
        bias = tf.Variable(tf.zeros(shape=shape_b), dtype=tf.float32)
        return weight, bias

    def linear_layer(self, x, num_out_channel):
        num_in_channel = x.get_shape().as_list()[-1]
        shape_w = [num_in_channel, num_out_channel]
        shape_b = [num_out_channel]
        w, b = self.hyper_initial(shape_w, shape_b)
        linear_out = tf.matmul(x, w)
        linear_out += b
        return linear_out

    def conv_layer(self, x, filter_size, num_filters, stride, actn=tf.nn.relu):
        num_in_channel = x.get_shape().as_list()[-1]
        shape_w = [filter_size, filter_size, num_in_channel, num_filters]
        shape_b = [num_filters]
        w, b = self.hyper_initial(shape_w, shape_b)
        layer = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        layer += b
        return actn(layer)

    def avg_pool(self, x, ksize, stride):
        pool_out = tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], \
                                  strides=[1, stride, stride, 1],\
                                  padding='SAME')    
        return pool_out

    def max_pool(self, x, ksize, stride):
        pool_out = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], \
                                  strides=[1, stride, stride, 1],\
                                  padding='SAME')    
        return pool_out

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    def fnn_layer(self, x, num_units, actn=tf.tanh, use_actn=False):
        in_dim = x.get_shape()[1]
        out_dim = num_units
        std = 0.01
        w = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std), dtype=tf.float32)
        b = tf.Variable(tf.zeros(shape=[1, out_dim]), dtype=tf.float32)
        A = tf.add(tf.matmul(x, w), b)
        if use_actn:
            A = actn(A)
        return A

