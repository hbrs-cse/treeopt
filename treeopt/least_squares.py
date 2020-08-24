import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize


class least_squares:
    """
    Python class, that bundles all modules nessesary to do least-squares optimization
    """

    def __init__(self):
        self.optimization_function_args = None
        self.diff_step = 0.01
        self.max_nfev = None

    def set_least_squares_function(self, function):
        self.optimization_function = function

    def set_least_squares_function_args(self, args):
        self.optimization_function_args = args

    def set_start_point(self, point):
        self.start_point = point

    def set_optimization_limits(self, limits):
        self.limits = limits

    def set_diff_step(self, value):
        self.diff_step = value
        
    def set_max_nfev(self, max_nfev):
        self.max_nfev = max_nfev

    def start_optimization(self):
        if self.optimization_function_args == None:
             op = optimize.least_squares(
                self.optimization_function,
                self.start_point,
                bounds=self.limits,
                diff_step = self.diff_step,
                max_nfev = self.max_nfev
            )
        else:
            op = optimize.least_squares(
                self.optimization_function,
                self.start_point,
                bounds=self.limits,
                args = self.optimization_function_args,
                diff_step = self.diff_step,
                max_nfev = self.max_nfev
            )

        self.sim_res = op
        return(op)
