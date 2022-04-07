import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa


def get_data(filename, ndata):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29
    r = 15
    s = int(((421 - 1) / r) + 1)

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, ::r, ::r].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, s * s)
    x = (x_branch, grid)
    y = y.reshape(ndata, s * s)
    return x, y


def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)


def main():
    x_train, y_train = get_data("piececonst_r421_N1024_smooth1.mat", 1000)
    x_test, y_test = get_data("piececonst_r421_N1024_smooth2.mat", 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    m = 29 ** 2
    activation = "relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((29, 29, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(128),
        ]
    )
    branch.summary()
    net = dde.maps.DeepONetCartesianProd(
        [m, branch], [2, 128, 128, 128, 128], activation, "Glorot normal"
    )

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)
    # net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    model.compile(
        tfa.optimizers.AdamW(1e-4, learning_rate=3e-4),
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=100000, batch_size=None)


if __name__ == "__main__":
    main()
