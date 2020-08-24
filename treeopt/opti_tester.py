# from scipy.optimize import minimize
# from treeopt.least_squares import least_squares

# def fun(x, a, b):
#     print(x)
#     return (a - x[0]) ** 2 + (b - x[1]) ** 2


# x0 = [1.3, 0.7]
# a = 5
# b = 3
# c = (a,b)
# res = minimize(fun, x0, args=(c), method="L-BFGS-B", tol=1e-5)
# print(res.x)

import scipy.optimize as optimize
import numpy as np


def parabola(params, x):
    a = params[0]
    b = params[1]
    c = params[2]

    y = a * (x - b) ** 2 + c

    return y


def error(params, *args):
    goal_parabola = args[0]
    x = args[1]

    op_parabola = parabola(params, x)

    sq_error = (op_parabola - goal_parabola) ** 2
    return sq_error


goal_params = (-3, 5, 2)
x_data = np.linspace(-5, 10, 100)

goal_parabola = parabola(goal_params, x_data)

my_args = (goal_parabola, x_data)

op = optimize.least_squares(
    error, (0, 0, 0), bounds=((-10, -10, -10), (10, 10, 10)), args=my_args
)

print(op.x)

