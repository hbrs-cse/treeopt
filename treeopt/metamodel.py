import smt.surrogate_models as smt


def rbf(xt, yt):
    """
    Function, that trains a metamodell based on the radial basis function
    algorithm
    :param xt: Array of points in which the system responce is known
    :type xt: Numpy array
    :param yt: Array contining the system responce
    :type yt: Numpy array
    :return: python object containing the trained metamodell
    :rtype: Smt-object
    """

    sm = smt.RBF(print_prediction=False, poly_degree=0, print_global=False)
    sm.set_training_values(xt, yt)
    sm.train()

    return sm


def krg(xt, yt):
    """
    Function, that trains a metamodell based on the kriging algorithm
    :param xt: Array of points in which the system responce is known
    :type xt: Numpy array
    :param yt: Array contining the system responce
    :type yt: Numpy array
    :return: python object containing the trained metamodell
    :rtype: Smt-object
    """

    sm = smt.KRG(theta0=[1e-2])
    sm.set_training_values(xt, yt)
    sm.train()

    return sm


def idw(xt, yt):
    """
    Function, that trains a metamodell based on the inverse distance weighting
    algorithm
    :param xt: Array of points in which the system responce is known
    :type xt: Numpy array
    :param yt: Array contining the system responce
    :type yt: Numpy array
    :return: python object containing the trained metamodell
    :rtype: Smt-object
    """

    sm = smt.IDW(p=2)
    sm.set_training_values(xt, yt)
    sm.train()

    return sm
