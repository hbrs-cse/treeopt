import numpy as np
import matplotlib.pyplot as plt

from treeopt import sampling
from treeopt import metamodell

import scipy.optimize as optimize


def problem(x):
    """
    Represents a two dimensional Function, with two local and one global minima
    :param x: 1D row-vector (multiple Values can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy array
    """

    y = -(
        np.exp(-((x - 2) ** 2))
        + np.exp(-((x - 6) ** 2) / 10)
        + 1 / (x ** 2 + 1)
    )

    return y


def calc_variances(x, sm):
    x = np.atleast_2d(x).reshape(x.shape[0], 1)
    var = sm.predict_variances(x)
    return var


def inverted_variance_function(x, sm):
    x = np.atleast_2d(x)
    ans = sm.predict_variances(x)
    return ans[0] * (-1)


def find_highest_variance(sm, limits, x_data):
    mi = []
    mif = []
    for x in np.linspace(limits[0][0], limits[0][1], 10):
        res = optimize.minimize(
            inverted_variance_function,
            x,
            args=sm,
            bounds=tuple(map(tuple, limits)),
            method="L-BFGS-B",
        )
        mi.append(res.x)
        mif.append(res.fun)

    min_val = None
    while min_val is None:
        min_val = mi[mif.index(min(mif))]
        if np.any(x_data == min_val):
            mi.pop(mif.index(min(mif)))
            mif.pop(mif.index(min(mif)))
            min_val = None

    return min_val, sm.predict_variances(np.atleast_2d(min_val))


def calc_sm_approximation(x, sm):
    x = np.atleast_2d(x)
    ans = sm.predict_values(x)
    return ans


def find_lowest_point(sm, limits, x_data):
    mi = []
    mif = []
    for x in np.linspace(limits[0][0], limits[0][1], 10):
        res = optimize.minimize(
            inverted_variance_function,
            x,
            args=sm,
            bounds=tuple(map(tuple, limits)),
            method="L-BFGS-B",
        )
        mi.append(res.x)
        mif.append(res.fun)

    min_val = None
    while min_val is None:
        min_val = mi[mif.index(min(mif))]
        if np.any(x_data == min_val):
            mi.pop(mif.index(min(mif)))
            mif.pop(mif.index(min(mif)))
            min_val = None

    return min_val


def calc_iters(maxiter, limits, n_doe):
    fig, axes = plt.subplots(nrows=maxiter, ncols=1)
    images = []

    x_data = sampling.full_factorial(limits, n_doe)
    y_data = problem(x_data[0])

    for i in range(1, x_data.shape[0]):
        y_data = np.vstack([y_data, problem(x_data[i])])

    vals = []

    for i in range(maxiter):
        x_space = np.linspace(limits[0][0], limits[0][1], 100)
        sm = metamodell.krg(x_data, y_data)

        pred_var = sm.predict_values(x_space)
        var = calc_variances(x_space, sm)

        x_n, val = find_highest_variance(sm, limits, x_data)
        vals.append(float(val))

        if vals[-1] / 0.20 < vals[0]:
            x_n = find_lowest_point(sm, limits, x_data)
            images.append(axes[i].plot(x_n, problem(x_n), "o", c="red"))
        else:
            images.append(axes[i].plot(x_n, problem(x_n), "x", c="red"))

        images.append(axes[i].plot(x_data, y_data, "xb"))
        images.append(axes[i].plot(x_space, problem(x_space), c="black"))
        images.append(axes[i].plot(x_space, pred_var, c="blue"))
        images.append(axes[i].plot(x_space, pred_var + var * 5, c="green"))
        images.append(axes[i].plot(x_space, pred_var - var * 5, c="green"))

        x_data = np.vstack([x_data, x_n])
        y_data = np.vstack([y_data, problem(x_n)])

    plt.show()
    return vals


limits = np.array([[-2, 10]])

vals = calc_iters(7, limits, 4)
