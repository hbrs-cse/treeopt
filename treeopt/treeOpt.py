import numpy as np
import os

# Experimental! imports to log errors
import traceback
import logging

from pathlib import Path

# Import of treeopt submodules
import treeopt.sampling as sampling
import treeopt.simulate as simulate
import treeopt.optimize as optimize
import treeopt.metamodell as metamodell
import treeopt.visualize as visualize


class TreeOpt:
    """
    Python class, that bundles all modules for nessesary for adaptive black box
    metamodelling
    """

    def __init__(self):
        """
        Pre allocates variables with default values
        :return: Nothing
        :rtype: None

        """

        self.samplingMethod = sampling.latin_hypercube
        self.smMethod = metamodell.krg
        self.problem_args = None
        self.sim_keyword = "extern"

    # Functions for Data management
    def append_x_data(self, xi):
        """
        Appends a datapoint xi to the design space
        :param xi: One point in the design space
        :type xi: Numpy array
        :return: Nothing
        :rtype: None

        """

        self.x = np.vstack([self.x, xi])

    def append_y_data(self, yi):
        """
        Appends a result yi to the result vector y
        :param yi: The result of one simultaion
        :type yi: Numpy array
        :return: nothing
        :rtype: none

        """

        self.y = np.vstack([self.y, yi])

    def write_data(self, npArray, filename):
        """sudo apt-get install pep8

        Function that writes a numpy Array into a file in the threeOptData
        direcory
        :param npArray: Array to be written onto the file
        :type npArray: Numpy array
        :param filename: Name of the file
        :type filename: String
        :return: Nothing
        :rtype: None

        """

        Path(os.getcwd() + "/treeOptData").mkdir(parents=True, exist_ok=True)
        path = os.path.join(os.getcwd() + "/treeOptData", filename + ".csv")
        np.savetxt(path, npArray, delimiter=",")

    def read_data(self, filename):
        """
        Function that reads a file in the treeOptData directory and creates a
        numpy array with this data
        :param filename: Name of the file to be read
        :type filename: String
        :return: Array containing the information of the file
        :rtype: Numpy Array

        """

        path = os.path.join(os.getcwd() + "/treeOptData", filename + ".csv")
        array = np.loadtxt(path, delimiter=",")
        return array

    # Set functions (to set functions/methods/variables prior to optimizaion)
    def set_name(self, name):
        """
        Sets a Variable with the name of the Optimization Problem
        :param name: the Name of the Optimization problem
        :type name: String
        :return: Nothing
        :rtype: None

        """

        self.name = name

    def set_sampling_method(self, method):
        """
        Sets the Sampling Method inteddet to be used by the individual
        Optimization Problem
        :param method: Python Object
        :type method: TYPE
        :return: Nothing
        :rtype: None

        """

        self.samplingMethod = method

    def set_num_doe(self, numDOE):
        """
        Sets the Number of Sampling Points to be used in the initial Sampling
        of the design space
        :param numDOE: Number of Sampling Points
        :type numDOE: Integer
        :return: Nothing
        :rtype: None

        """

        self.numDOE = numDOE

    def set_limits(self, limits):
        """
        Sets the search limits of each dimension of the design space
        :param limits: Array containing the highest and lowest limits for each
        dimension
        :type limits: Numpy-Array
        :return: Nothing
        :rtype: None

        """

        self.limits = limits

    def set_problem(self, problem):
        """
        Sets the problem of which an optimization is to be done. Arguments have
        to be passed in the following way:
            function(x, arg1, arg2, ..., argN), where x are the parameters
            which are to be optimized and arg1 to argN are static Variables.
        :param problem: A function that takes takes parameters and returnes the
        system responce
        :type problem: Python function
        :return: Nothing
        :rtype: None

        """
        self.problem = problem

    def set_problem_args(self, args):
        """
        Sets additional Arguments, which are to be passed with each call of the
        problem function. Arguments have to be passed in the same order in
        which the they are declared in the problem definition
        :param args: Tuple containing the arguments
        :type args: Tuple
        :return: Nothing
        :rtype: None

        """

        self.problem_args = args

    def set_sim_keyword(self, keyword):
        """
        Adds a keyword to the Optimization Algorithm. When a MEtamodell is to
        be visualized, the value of the keyword is checked and depending on the
        keyword a different plot is generated
        :param keyword: Keyword describing the problem-type (default "extern",
        other alternative "benchmark")
        :type keyword: String
        :return: Nothing
        :rtype: None

        """

        self.sim_keyword = keyword

    def set_sm_method(self, method):
        """
        Sets the Method that is used to approximate the system responce in the
        design space
        :param method: one of the functions in modules/metamodell.py
        :type method: python function
        :return: Nothing
        :rtype: None

        """

        self.smMethod = method

    def set_accuracy_criterion(self, method):
        """
        Sets the method that is used to define the termination condition of the
        optimization workflow
        :param method: one of the functions in modules/accuracy
        :type method: Python Function
        :return: Nothing
        :rtype: None

        """

        self.accuracyCriterion = method

    def simulate_problem(self, x):
        if self.problem_args == None:
            return self.problem(x)
        else:
            return self.problem(x, self.problem_args)

    def start_optimization(self):
        """
        Function that starts the previosly parameterized adaptive optimization
        loop
        :return: Nothing
        :rtype: None

        """

        self.x = self.samplingMethod(self.limits, self.numDOE)

        self.y = self.simulate_problem(np.atleast_2d(self.x[0]))
        for xi in self.x[1:]:
            print(self.simulate_problem(np.atleast_2d(xi)))
            self.append_y_data(self.simulate_problem(np.atleast_2d(xi)))

        self.write_data(self.x, "DoeData")
        self.write_data(self.y, "DoeResponce")

        self.optGoal = False

        self.ite = 0
        self.maxite = 5

        while self.optGoal is False:

            self.sm = self.smMethod(self.x, self.y)

            self.nX = optimize.get_lowest_variance(self.sm, self.limits)

            self.append_x_data(self.nX)
            self.append_y_data(self.simulate_problem(np.atleast_2d(self.nX)))

            self.write_data(self.x, "DoeData")
            self.write_data(self.y, "DoeResponce")

            self.ite += 1

            if self.ite == self.maxite:
                self.optGoal = True

            self.current_best_point = optimize.find_minimum(
                self.sm, self.limits
            )
            print("bester Punkt:", self.current_best_point)

        vis = visualize.Visualize(self)
        vis.plot()
