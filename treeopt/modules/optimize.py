import numpy as np
import scipy.optimize as optimize


def get_largest_uncertainty_function(sm):
    """
    Defines and returns a python object, that is able to calculate the assumed
    uncertainty of a metamodell
    :param sm: A metamodell object from the smt-toolkit. Must be able to
    calculate variances
    :type sm: Smt-object
    :return: object that when called can return the uncertainty of a point
    :rtype: Python-function

    """

    def un(x):
        """
        Function that calculates the uncertainty of a metamodell at a given
        point
        :param x: Numpy array representing a point in the designspace
        :type x: Numpy-Array
        :return: Absoulte-value of the uncertainty at one specific point
        :rtype: Numpy-Array

        """

        x = np.atleast_2d(x)

        sm_val = sm.predict_values(x)
        sm_var = sm.predict_variances(x)

        pyvar = sm_val + 3 * np.sqrt(sm_var)
        nyvar = sm_val - 3 * np.sqrt(sm_var)

        return np.abs(pyvar - nyvar)

    return un


def get_lowest_variance_function(sm):
    """
    Defines and returns a python object, that is able to calculate the lowest
    assumed variance of a metamodell
    :param sm: A metamodell object from the smt-toolkit. Must be able to
    calculate variances
    :type sm: Smt-object
    :return: Object that when called returns the lower variance of a point
    :rtype: Python-function

    """

    def mi(x):
        """
        Function that calculates the lower variance of a metamodell at one
        point
        :param x: Numpy array representing a point in the designspace
        :type x: Numpy array
        :return: Lover variance of the metamodell one specific point
        :rtype: Numpy array

        """

        x = np.atleast_2d(x)

        sm_val = sm.predict_values(x)
        sm_var = sm.predict_variances(x)

        nyvar = sm_val - 3 * np.sqrt(sm_var)

        return nyvar[0]

    return mi


def get_minimum_function(sm):
    """
    Defines and returns a python object, that returns the value of the responce
    surface at a given point
    :param sm: A metamodell object from the smt-toolkit.
    :type sm: Smt-object
    :return: Object that when called returns the functon value of the
    metamodell
    :rtype: Python-function

    """

    def metamodell_value_function(x):
        """
        Function that returns the value of the responce surface at a given
        point
        :param x: Numpy array representing a point in the designspace
        :type x: Numpy array
        :return: Value of the responce surface at the point x
        :rtype: Numpy array

        """

        x = np.atleast_2d(x)

        y = sm.predict_values(x)

        return y[0]

    return metamodell_value_function


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

    unc_fun = get_largest_uncertainty_function(sm)

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]

    res = optimize.minimize(
        unc_fun, start, bounds=tuple(map(tuple, limits)), method="L-BFGS-B"
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

    min_var = get_lowest_variance_function(sm)

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]

    res = optimize.minimize(
        min_var, start, bounds=tuple(map(tuple, limits)), method="L-BFGS-B"
    )

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

    min_fun = get_minimum_function(sm)

    start = np.empty(limits.shape[0], dtype=float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]

    res = optimize.minimize(
        min_fun, start, bounds=tuple(map(tuple, limits)), method="L-BFGS-B"
    )

    return res.x
