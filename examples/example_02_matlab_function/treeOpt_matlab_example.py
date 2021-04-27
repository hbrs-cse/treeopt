from treeopt.treeOpt import adaptive_metamodell
import numpy as np
import os

def execute_matlab():
    os.system("matlab -nodesktop -nosplash -batch matlab_function")

def read_matlab_output():
    data = np.loadtxt("output.txt")
    return(data)

def write_matlab_input(data):
    np.savetxt("input.txt", data)

def simulation_procedure(x):
    write_matlab_input(x)
    execute_matlab()
    ret = read_matlab_output()
    return(ret)

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
fun = simulation_procedure
opt.set_cost_function(fun)

# Sets the keyword to plot the model
opt.set_vis_keyword("extern")

# Starts the Optimization
opt.optimize()
