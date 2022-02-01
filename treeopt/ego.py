import numpy as np

from smt.applications import EGO
from smt.surrogate_models import KRG
import scipy.optimize as sciopt

from treeopt.visualize2 import VisualizeMetamodel

import matplotlib.pyplot as plt

class OptimizeEGO:
    """
    Class that runs a Efficient Global Optimization Algorithm on a metamodel. The number of iterations is not fixed.
    The algorithm runs until one of two possible termination criteria is fulfilled. The first criterion checks, if two
    support points of the metamodel are closer than a minimum threshold. If that state is not archived prior before a
    maximum of allowed function evaluations is reached, the optimization also terminates.
    """
    def __init__(self):
        self._doe_data = None
        self._x_data = []
        self._y_data = []
        self._opti_object = None
        self._x_limits = None
        self._num_evals = 0
        self._max_evals = None
        self._x_opt = None
        self._y_opt = None

    @property
    def doe_data(self):
        """
        Getter for the self._doe_data variable. This variable stores the datapoints that are part of the first support
        points of the metamodell.
        """
        return self._doe_data

    @doe_data.setter
    def doe_data(self, data):
        """
        Setter for the self._doe_data variable
        """
        self._doe_data = data

    @property
    def x_data(self):
        """
        Getter for the self._x_data variable. The variable stores the input variables that have been simulated.
        """
        return self._x_data

    @x_data.setter
    def x_data(self, data):
        """
        Setter for the self._x_data variable. If the variable is empty, the variable is overwritten. In all other cases
        the data is appended to the list.
        """
        if self._x_data is None:
            self._x_data = data
        else:
            self._x_data.append(data)

    @property
    def y_data(self):
        """
        Getter for the self._y_data variable. the variable stores the output variables that have been simulated.
        """
        return self._y_data

    @y_data.setter
    def y_data(self, data):
        """
        Setter for the self._y_data variable. If the variable is empty, the variable is overwritten. In all other cases
        the data is appended to the list.
        """
        if self._y_data is None:
            self.y_data = data
        else:
            self._y_data.append(data)

    @property
    def opti_object(self):
        """
        Getter for the self._opti_object variable. The variable manages the an object that returns a results value that
        corresponds to the given input value.
        """
        return self._opti_object

    @opti_object.setter
    def opti_object(self, object):
        """
        Setter for the self._opti_object variable.
        """
        self._opti_object = object

    @property
    def x_limits(self):
        """
        Getter for the self._x_limits variable. The variable stores the limits of the design space.
        """
        return self._x_limits

    @x_limits.setter
    def x_limits(self, limits):
        """
        Setter for the self._x_limits variable
        """
        self._x_limits = limits

    @property
    def max_evals(self):
        """
        Getter for the self._max_evals variable. This variable stores the number of function evaluations. This value of
        this variable continuously gets compared to the maximum number of allowed function evaluations.
        """
        return self._max_evals

    @max_evals.setter
    def max_evals(self, number):
        """
        Setter for the self._max_evals variable
        """
        self._max_evals = number

    @property
    def x_opt(self):
        """
        Getter for the self._x_opt variable. The variable stores the current best input data
        """
        return self._x_opt_ego

    @x_opt.setter
    def x_opt(self, opt):
        """
        Setter for the self._x_opt variable
        """
        self._x_opt_ego = opt

    @property
    def y_opt(self):
        """
        Getter for the self._y_opt variable. The variable stores the current best simulation result data.
        """
        return self._y_opt

    @y_opt.setter
    def y_opt(self, opt):
        """
        Setter for the self._y_opt variable
        """
        self._y_opt = opt

    @property
    def num_evals(self):
        """
        Getter for the self._num_evals variable. The variable stores a number that gets increased by one each time the
        objective function gets evaluated.
        """
        return self._num_evals

    @num_evals.setter
    def num_evals(self, num):
        """
        Setter for the self._num_evals variable.
        """
        self._num_evals = num

    def get_datapoints(self):
        """
        Function that returns disappoints to the metamodel. For the first iteration the date from the self.doe_data
        """
        if self.x_data == []:
            return self.doe_data
        else:
            return np.array(self.x_data)

    def calc_min_distance(self):
        """
        Function that returns the distance of the two closest points in the self.x_data variable.
        """
        data = []

        for i_num, prim_element in enumerate(self.x_data):
            for sec_element in self.x_data[i_num + 1:]:
                data.append(
                    np.linalg.norm(
                        np.abs(np.array(prim_element) - np.array(sec_element))
                    )
                )

        return min(data)

    def solve_problem(self, design_vector):
        """
        Function that returns the simulation result for a given design vector. If the value is already known, the
        simulation is not started but the value from the self.y_data variable gets returned.
        """
        ret = []
        for datapoint in design_vector.tolist():
            if datapoint in self.x_data:
                ret.append(self.y_data[self.x_data.index(datapoint)])
            else:
                result = self.opti_object(datapoint)
                self.x_data = datapoint
                self.y_data = result
                self.num_evals += 1
                ret.append(result)

        return np.array(ret)

    def do_ego(self, min_dist):
        """
        Starts the EGO algorithm for the next iteration.
        """
        done = False
        while not done:
            ego = EGO(
                n_iter=1,
                criterion="EI",
                n_start=50,
                xdoe=self.get_datapoints(),
                xlimits=self.x_limits,
            )
            self.x_opt, self.y_opt, _, x_data, y_data = ego.optimize(fun=self.solve_problem)

            if self.calc_min_distance() < min_dist or self.num_evals == self.max_evals:
                done = True

    def conv_plot(self):
        """
        Uses the data stored in self.y_data to create a convergence plot.
        """

        lowest = self.y_data[0]
        li = [lowest]
        for number in self.y_data[1:]:
            if number < lowest:
                lowest = number
            li.append(lowest)

        plt.plot(self.y_data, "x")
        plt.plot(li, "r-")
        plt.yscale("log")
        plt.xlabel("Funktionsauswertung")
        plt.ylabel("Wert der Zielfunktion")
        plt.xlim([0,len(self.y_data)])
        plt.grid()

        plt.show()

    def plot_design_space(self):
        """
        Function that plots the simulated points in the design space. Only works for functions with a two dimensional
        design space.
        """
        data = np.array(self.x_data)
        len_doe = len(self._doe_data)

        plt.plot(data[:, 0][:len_doe], data[:, 1][:len_doe], "rx")
        plt.plot(data[:, 0][len_doe:], data[:, 1][len_doe:], "bx")
        plt.plot(self.x_opt[0], self.x_opt[1], "gx")

        plt.grid()
        plt.xlim(self.x_limits[0])
        plt.ylim(self.x_limits[1])

        plt.xticks(np.arange(self.x_limits[0][0], self.x_limits[0][1], 1))
        plt.yticks(np.arange(self.x_limits[1][0], self.x_limits[1][1], 1))

        plt.xlim(self.x_limits[0])
        plt.ylim(self.x_limits[1])

        plt.show()

    def visualize(self):
        """
        Visualizes the created metamodell using the VisualizeMetamodel class.
        """
        sm = KRG(theta0=[1e-2])
        sm.set_training_values(np.array(self.x_data), np.array(self.y_data))
        sm.train()

        vis = VisualizeMetamodel()
        vis.bounds = self.x_limits
        vis.metamodel = sm
        vis.points_per_axis = 100
        vis.num_doe_points = len(self.doe_data)

        vis.plot_metamodel()

