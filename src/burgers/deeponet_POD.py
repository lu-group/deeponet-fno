import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import io

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa


def get_data(ntrain, ntest):
    sub_x = 2 ** 6
    sub_y = 2 ** 6

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat("burgers_data_R10.mat")
    x_data = data["a"][:, ::sub_x].astype(np.float32)
    y_data = data["u"][:, ::sub_y].astype(np.float32)
    x_branch_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_branch_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    s = 2 ** 13 // sub_y  # total grid size divided by the subsampling rate
    grid = np.linspace(0, 1, num=2 ** 13)[::sub_y, None]

    x_train = (x_branch_train, grid)
    x_test = (x_branch_test, grid)
    return x_train, y_train, x_test, y_test


def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    # w_cumsum = np.cumsum(w)
    # print(w_cumsum[:16] / w_cumsum[-1])
    # plt.figure()
    # plt.plot(y_mean)
    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.plot(v[:, i])
    # plt.show()
    return y_mean, v


class PODDeepONet(dde.maps.NN):
    def __init__(
        self,
        pod_basis,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(
                activation
            )

        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float32)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = dde.maps.FNN(
                layer_sizes_branch, activation_branch, kernel_initializer
            )
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = dde.maps.FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        x_func = self.branch(x_func)
        if self.trunk is None:
            # POD only
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


def main():
    x_train, y_train, x_test, y_test = get_data(1000, 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    y_mean, v = pod(y_test)
    modes = 32
    m = 2 ** 7
    net = PODDeepONet(v[:, :modes], [m, 128, 128, modes], None, "tanh", "Glorot normal")

    def output_transform(inputs, outputs):
        # return outputs + y_mean
        # return outputs / modes ** 0.5 + y_mean
        return outputs / modes + y_mean

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=3e-4,
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=500000, batch_size=None)


if __name__ == "__main__":
    main()
