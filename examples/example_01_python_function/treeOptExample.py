from treeopt.treeOpt import adaptive_metamodell
import numpy as np

def himmelblaus_function(x, *args):
    y = (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
    return y

# Generates a TreeOpt object
opt = adaptive_metamodell()

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





