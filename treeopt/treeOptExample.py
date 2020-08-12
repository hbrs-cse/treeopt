from treeOpt import TreeOpt
import modules.benchmark as benchmark

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

opt.set_simulate_method("benchmarking")

# Gets a benchmarking function
fun = benchmark.himmelblaus_function
opt.set_python_problem(fun)

# Starts the Optimization
opt.start_optimization()
