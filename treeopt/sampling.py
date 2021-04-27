import smt.sampling_methods as smtSam


def latin_hypercube(limits, ndoe):
    """
    Function to define the experiment by using a latin hypercube algorithm
    :param limits: Numpy array representing the limit of the desingnspace
    :type limits: Numpy array
    :param ndoe: Number of points to be sampled in the designspace
    :type ndoe: Integer
    :return: Numpy array containg the sampled points
    :rtype: Numpy array
    """

    sampling = smtSam.LHS(xlimits=limits, criterion="m")
    x = sampling(ndoe)
    return x


def full_factorial(limits, ndoe):
    """
    Function to define the experiment by using full factorial grid. If
    :param limits: numpy array representing the limit of the desingnspace
    :type limits: Numpy array
    :param ndoe: Number of points to be sampled in the designspace
    :type ndoe: Integer
    :return: Numpy-array containg the sampled points
    :rtype: Numpy array
    """

    sampling = smtSam.FullFactorial(xlimits=limits)
    x = sampling(ndoe)
    return x


def random(limits, ndoe):
    """
    Function define the experiment by randomly picking points in the
    designspace
    :param limits: Numpy array representing the limit of the desingnspace
    :type limits: Numpy array
    :param ndoe: Number of points to be sampled in the designspace
    :type ndoe: Integer
    :return: Numpy-array containg the sampled points
    :rtype: Numpy array
    """

    sampling = smtSam.Random(xlimits=limits)
    x = sampling(ndoe)
    return x
