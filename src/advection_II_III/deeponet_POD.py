import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf


def get_data(filename):
    nx = 40
    nt = 40
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, nt * nx)
    return (u0, xt), u


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
    nt = 40
    nx = 40
    x_train, y_train = get_data("train_IC2.npz")
    x_test, y_test = get_data("test_IC2.npz")
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    pca = PCA(n_components=0.9999).fit(y_train)
    print("# Components:", pca.n_components_)
    # print(np.cumsum(pca.explained_variance_ratio_))
    # plt.figure()
    # plt.semilogy(pca.explained_variance_ratio_, 'o')
    # plt.figure()
    # plt.imshow(pca.mean_.reshape(nt, nx))
    # plt.colorbar()
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(pca.components_[i].reshape(nt, nx) * 40)
    #     plt.colorbar()
    # plt.show()
    net = PODDeepONet(
        pca.components_.T * 40,
        [nx, 512, pca.n_components_],
        None,
        "relu",
        "Glorot normal",
    )

    def output_transform(inputs, outputs):
        return outputs / pca.n_components_ + pca.mean_

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=100000, batch_size=None)


if __name__ == "__main__":
    main()
