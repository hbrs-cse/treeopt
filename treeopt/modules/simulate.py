def simulateBenchmarkFunction(fun, x):
    """Returns the function value of a benchmarking function
    INPUT:
        fun: Python object representing the benchmarking function
        x: Numpy-Array representing a point on which the benchmarkin function is to be evaluated
    OUTPUT:
        function Value of the function at the point x
    """
    return(fun(x))