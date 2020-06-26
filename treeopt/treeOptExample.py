from treeOpt import TreeOpt
from treeOpt import benchmark

import numpy as np

#Generates a TreeOpt object
opt = TreeOpt()

#Sets a name
opt.setName("Benchmarking-problem")

#Sets Limis of the TreeOpt Object
limits = np.array([[-5,5],[-5,5]])
opt.setLimits(limits)

#Sets number of point in the initial sampling apace
ndoe = 15
opt.setNumDOE(ndoe)

#Gets a benchmarking function
fun = benchmark.HimmelblausFunction
opt.setBenchmarkingProblem(fun)

#Starts the Optimization
opt.startOptimization()
