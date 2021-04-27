from treeopt.treeOpt import TreeOpt
import benchmark
import numpy as np

# Generates a TreeOpt object
opt = TreeOpt()

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
fun = benchmark.himmelblaus_function
opt.set_problem(fun)

# Starts the Optimization
opt.start_optimization()
