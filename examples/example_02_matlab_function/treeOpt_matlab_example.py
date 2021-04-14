from treeopt.treeOpt import adaptive_metamodell
from treeopt import simulate
import numpy as np


def calc_matlab(x):
    inputFile = "input.txt"
    outputFile = "output.txt"
    simFile = "matlab_function.m"
    program = "matlab"
    responce = simulate.simulate_external_programm(
        inputFile, outputFile, simFile, program, x
    )

    return responce


# Generates a TreeOpt object
opt = adaptive_metamodell()

# Sets a name
opt.set_name("TreeOpt-Matlab-Example")

# Sets Limis of the TreeOpt Object
limits = np.array([[-5, 5]])
opt.set_limits(limits)

# Sets number of point in the initial sampling apace
ndoe = 5
opt.set_num_doe(ndoe)

# Gets a benchmarking function
fun = calc_matlab
opt.set_problem(fun)

# Starts the Optimization
opt.start_optimization()