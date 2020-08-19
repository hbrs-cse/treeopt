from scipy.optimize import minimize


def fun(x, a, b):
    return (a - x[0]) ** 2 + (b - x[1]) ** 2


x0 = [1.3, 0.7]
a = 5
b = 3
res = minimize(fun, x0, args=(a, b), method="L-BFGS-B", tol=1e-5)
print(res.x)
