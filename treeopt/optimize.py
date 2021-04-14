import numpy as np
import scipy.optimize as optimize
import time


def get_largest_uncertainty_function(x, sm):
    """
    Function to calculate and return a the function value of the uncertainty at
    a given point x
    :param sm: Python object representing the benchmarking function
    :type sm: SMT-Object
    :param x: Numpy array representing a point on which the lowest varinace
    function is to be evaluated
    :type x: Numpy array
    :return: Function Value at the point x
    :rtype: Numpy array
    """

    x = np.atleast_2d(x)

    sm_val = sm.predict_values(x)
    sm_var = sm.predict_variances(x)

    pyvar = sm_val + 3 * np.sqrt(sm_var)
    nyvar = sm_val - 3 * np.sqrt(sm_var)

    return np.abs(pyvar - nyvar)


def get_lowest_variance_function(x, sm):
    """
    Function to calculate the lower variance of an SMT-metamodell at a given
    point x
    :param sm: Python object representing the benchmarking function
    :type sm: SMT-Object
    :param x: Numpy array representing a point on which the lowest varinace
    function is to be evaluated
    :type x: Numpy array
    :return: Function Value at the point x
    :rtype: Numpy array
    """

    x = np.atleast_2d(x)

    sm_val = sm.predict_values(x)
    sm_var = sm.predict_variances(x)

    nyvar = sm_val - 3 * np.sqrt(sm_var)

    return nyvar[0]


def get_minimum_function(x, sm):
    """
    Returns the approximation of a function value in of the matamodell
    :param sm: Python object representing the benchmarking function
    :type sm: SMT-Object
    :param x: Numpy array representing a point on which the lowest varinace
    function is to be evaluated
    :type x: Numpy array
    :return: Function Value at the point x
    :rtype: Numpy array
    """
    x = np.atleast_2d(x)

    y = sm.predict_values(x)

    return y[0]


def get_higest_uncertainty(sm, limits):
    """
    Searches in the responce surface of a given metamodell for the point in the
    designspace with the highest uncertaintiy
    :param sm: A metamodell object from the smt-toolkit. Must be able to
    calculate variances
    :type sm: Smt-object
    :param limits: Limits of the designspace
    :type limits: Numpy-array
    :return: Point in the design space with the highest uncertainty
    :rtype: Numpy-array

    """

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]

    res = optimize.minimize(
        get_largest_uncertainty_function,
        start,
        args=(sm),
        bounds=tuple(map(tuple, limits)),
        method="L-BFGS-B",
    )

    return res.x


def get_lowest_variance(sm, limits):
    """
    Searches in the responce surface of a given metamodell for the point in
    the designspace with the lowest variance
    :param sm: A metamodell object from the smt-toolkit. Must be able to
    calculate variances
    :type sm: Smt-object
    :param limits: Limits of the designspace
    :type limits: Numpy-array
    :return: Point in the design space with the highest uncertainty
    :rtype: Numpy-array

    """

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = (limits[dim][1] + limits[dim][0]) / 2

    res = optimize.minimize(
        get_lowest_variance_function,
        start,
        args=(sm),
        bounds=tuple(map(tuple, limits)),
        method="L-BFGS-B",
    )

    print(res)

    return res.x


def find_minimum(sm, limits):
    """
    Searches in the responce surface of a given metamodell for the Point with
    the lovest valued system responce
    :param sm: DESCRIPTION
    :type sm: TYPE
    :param limits: DESCRIPTION
    :type limits: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]

    res = optimize.minimize(
        get_minimum_function,
        start,
        args=(sm),
        bounds=tuple(map(tuple, limits)),
        method="L-BFGS-B",
    )

    return res.x
