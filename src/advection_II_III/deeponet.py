import sys

import numpy as np
import matplotlib.pyplot as plt
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


def main():
    nt = 40
    nx = 40
    x_train, y_train = get_data("train_IC2.npz")
    x_test, y_test = get_data("test_IC2.npz")
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    # IC1
    # losshistory, train_state = model.train(epochs=100000, batch_size=None)
    # IC2
    losshistory, train_state = model.train(epochs=250000, batch_size=None)

    y_pred = model.predict(data.test_x)
    np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
    np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
    np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))


if __name__ == "__main__":
    main()
