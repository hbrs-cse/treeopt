from treeopt.treeOpt import TreeOpt
import numpy as np

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

# Generates a TreeOpt object
opt = TreeOpt()

# Sets a name
opt.set_name("Benchmarking-problem")

# Sets Limis of the TreeOpt Object
limits = np.array([[-5, 5],[-9, 5],[6,9]])
opt.set_limits(limits)

# Sets number of point in the initial sampling apace
ndoe = 5
opt.set_num_doe(ndoe)

# Gets a benchmarking function
opt.set_sim_keyword("extern")
fun = rotated_hyper_ellipsoid_function
opt.set_problem(fun)

# Starts the Optimization
opt.start_optimization()