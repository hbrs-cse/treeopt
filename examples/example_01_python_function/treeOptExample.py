from treeopt.treeOpt import adaptive_metamodell
import numpy as np

def himmelblaus_function(x, *args):
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

# Generates a TreeOpt object
opt = adaptive_metamodell()

# Sets a name
opt.set_name("Benchmarking-problem")

# Sets Limis of the TreeOpt Object
limits = np.array([[-5, 5], [-5, 5]])
opt.set_limits(limits)

# Sets number of point in the initial sampling apace
ndoe = 15
opt.set_num_doe(ndoe)

# Gets a benchmarking function
opt.set_sim_keyword("benchmarking")
fun = himmelblaus_function
opt.set_problem(fun)

# Starts the Optimization
opt.start_optimization()
