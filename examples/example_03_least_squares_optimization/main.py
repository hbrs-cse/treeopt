import numpy as np
from treeopt.treeOpt import least_squares

import matplotlib.pyplot as plt

def parabola(params, x):
    """
    Returns the function-values described by a parabola with the passed
    parameters. The evaluations occur at the points in x.
    :param params: Parameters to describe the parabola
    :type params: Numpy array
    :param x: DESCRIPTION
    :type x: Numpy array
    :return: The function values evaluated at x
    :rtype: Numpy array

    """

    a = params[0]
    b = params[1]
    c = params[2]

    y = a * (x - b) ** 2 + c

    return y


def error(params, *args):
    """
    Calculates the quadratic error of the quess of the optimization and the
    optimization goal
    :param params: Parameters to describe the parabola
    :type params: Numpy array
    :param *args: Additional arguments that. Here args[0] is the optimization
    goal and args[1] are the points where the comparisons are to be made
    :type *args: Numpy array
    :return: Quadratic error of the compared arrays
    :rtype: Numpy array

    """

    goal_parabola = args[0]
    x = args[1]

    op_parabola = parabola(params, x)

    sq_error = (op_parabola - goal_parabola) ** 2

    return sq_error


# Define Points on which function Evaluations are to be made
x_data = np.linspace(-5, 10, 100)

# Define the optimization goal
goal_params = (-3, 5, 2)
goal_parabola = parabola(goal_params, x_data)

# Generate an least_squares object from the treeopt class
op = least_squares()

# Sets the error function as the function to be analyzed
op.set_cost_function(error)

# Sets a starting point for the optimization
start_point = (0, 0, 0)
op.set_start_point(start_point)

# Sets upper and lower Bounds for the variables
bounds = ((-10, -10, -10), (10, 10, 10))
op.set_limits(bounds)

# Sets additional arguments to be passed to the function
my_args = (goal_parabola, x_data)
op.set_cost_function_args(my_args)

# Initializes the Optimization
res = op.optimize()

# Prints the Result of the optimization
print(res.nfev)
print(res.x)

found_parabola = parabola(res.x, x_data)

# plt.plot(x_data, goal_parabola)
# plt.plot(x_data, found_parabola)
plt.plot(x_data, goal_parabola-found_parabola)
plt.plot(x_data, (goal_parabola-found_parabola)**2)
plt.grid()
plt.show()