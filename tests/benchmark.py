import numpy as np


def tree_valley_function(x):
    """
    Represents a two dimensional Function, with two local and one global minima
    :param x: 1D Rowvector (multiple Values can be passed)
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


def four_humps_function(x):
    """
    Function with one global maxima, one local maximum, one glomal minimun an
    a locam minimum
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy array
    """

    y = (1 - x[:, 0] / 2 + x[:, 0] ** 5 + x[:, 1] ** 3) * np.exp(
        -x[:, 0] ** 2 - x[:, 1] ** 2
    )

    return y


def himmelblaus_function(x):
    """
    The Himmelblau Function is a multi-modal function, with one local maximum
    four local minima.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy array
    """

    y = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2

    return y


def rotated_hyper_ellipsoid_function(x):
    """
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal.
    It is an extension of the Axis Parallel Hyper-Ellipsoid function,
    also referred to as the Sum Squares function.
    :param x: N-Dimensional Vector rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy Array
    """
    x = np.atleast_2d(x)
    y = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        y = y + x[:, i] ** 2

    return y


def matyas_function(x):
    """
    The Matyas function has no local minima except the global one.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy array
    """

    y = 0.26 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 0.48 * x[:, 0] * x[:, 1]

    return y


def cross_in_tray_function(x):
    """
    The Cross-in-Tray function has multiple global minima. It is shown here
    with a smaller domain in the second plot, so that its characteristic
    "cross" will be visible.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function values of x
    :rtype: Numpy array
    """

    fact1 = np.sin(x[:, 0]) * np.sin(x[:, 1])
    fact2 = np.exp(np.abs(100 - np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi))
    y = -0.0001 * (abs(fact1 * fact2) + 1) ** 0.1
    return y


def SixHumpCamelFunction(x):
    """
    he plot on the left shows the six-hump Camel function on its recommended
    input domain, and the plot on the right shows only a portion of this
    domain, to allow for easier viewing of the function's key characteristics.
    The function has six local minima, two of which are global.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: function values of x
    :rtype: Numpy array
    """

    x1 = x[:, 0]
    x2 = x[:, 1]
    y = (
        (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
        + x1 * x2
        + (-4 + 4 * x2 ** 2) * x2 ** 2
    )
    return y


def michalewicz_function(x):
    """
    The Michalewicz function has d! local minima, and it is multimodal.
    The parameter m defines the steepness of they valleys and ridges;
    a larger m leads to a more difficult search.
    The recommended value of m is m = 10.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function Values of x
    :rtype: Numpy array
    Funktioniert nicht so wie es soll
    Quelle:
        http://www.sfu.ca/~ssurjano/michal.html
    """

    m = 10
    y = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        y = y + np.sin(x[:, i]) * (np.sin(i * x[:, i] ** 2 / np.pi)) ** (2 * m)
    return -y


def rastrigin_function(x):
    """
    The Rastrigin function has several local minima. It is highly multimodal,
    but locations of the minima are regularly distributed.
    :param x: 2D rowvector (a column of rows can be passed)
    :type x: Numpy array
    :return: Function Values of x
    :rtype: Numpy array
    """

    dim = x.shape[1]
    y = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        y = y + x[:, i] - 10 * np.cos(2 * np.pi * x[:, i])
    return 10 * dim + y
