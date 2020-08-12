import numpy as np
import os

# Experimental! imports to log errors
import traceback
import logging

# Import of treeopt submodules
import modules.sampling as sampling
import modules.simulate as simulate
import modules.optimize as optimize
import modules.benchmark as benchmark
import modules.metamodell as metamodell
import modules.visualize as visualize


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
        self.simulateMethod = simulate.simulate_benchmark_function
        self.smMethod = metamodell.krg

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

    def set_simulate_method(self, method):
        """
        Sets the method, which should be used to start the Simulation
        :param method: A keyword representing the method
        the simulation String
        :type method: Python function
        :return: Nothing
        :rtype: None

        """
        self.simKeyword = method
        if self.simKeyword == "benchmarking":
            self.simulateMethod = simulate.simulate_benchmark_function
        elif self.simKeyword == "python":
            self.simulateMethod = simulate.simulate_benchmark_function
        elif self.simKeyword == "extern":
            self.simulateMethod = simulate.simulateExternalProgramm
        else:
            print("Error")

    def set_python_problem(self, problem):
        """
        Stores a Function, that represents a benchmarking problem
        :param problem: Python function, which returns points of a responce
        surface
        :type problem: Python function
        :return: Nothing
        :rtype: None

        """

        self.benchmarkingProblem = problem

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

    # Starts one simulation run
    def simulate(self, x):
        """
        Function that starts a Simulation and returns the resonce of the system
        :param x: Numpy array representing a set of parameters to be written on
        the simulation Input file
        :type x: Numpy array
        :return: Numpy array containing the system responce
        :rtype: Numpy array

        """

        # if self.simulateMethod == simulate.simulate_benchmark_function:
        #     return self.simulateMethod(self.benchmarkingProblem, x)
        # else:
        #     print("Fehler")

        try:
            if self.simulateMethod == simulate.simulate_benchmark_function:
                return self.simulateMethod(self.benchmarkingProblem, x)
            # TODO if the problem is not a benchmarking problem, a simulation
            # is to be started here.
        except:
            logging.error(traceback.format_exc())

    def start_optimization(self):
        """
        Function that starts the previosly parameterized adaptive optimization
        loop
        :return: Nothing
        :rtype: None

        """

        self.x = self.samplingMethod(self.limits, self.numDOE)
        self.y = self.simulate(np.atleast_2d(self.x[0]))

        for xi in self.x[1:]:
            self.append_y_data(self.simulate(np.atleast_2d(xi)))

        self.write_data(self.x, "DoeData")
        self.write_data(self.y, "DoeResponce")

        self.optGoal = False

        self.ite = 0
        self.maxite = 5

        while self.optGoal is False:

            self.sm = self.smMethod(self.x, self.y)

            self.nX = optimize.get_lowest_variance(self.sm, self.limits)

            self.append_x_data(self.nX)
            self.append_y_data(self.simulate(np.atleast_2d(self.nX)))

            self.write_data(self.x, "DoeData")
            self.write_data(self.y, "DoeResponce")

            self.ite += 1

            if self.ite == self.maxite:
                self.optGoal = True

            self.current_best_point = optimize.find_minimum(
                self.sm, self.limits
            )
            print("bester Punkt:", self.current_best_point)

        vis = visualize.visualize(self)
        vis.plot()
