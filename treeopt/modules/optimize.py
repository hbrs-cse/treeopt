import numpy as np

from scipy.optimize import basinhopping
from scipy.optimize import fmin_l_bfgs_b

def getLargestUncertaintyFunction(sm):
    """defines a python object, that is able to calculate the assumed uncertainty of a metamodell
    INPUT:
        sm: a metamodell object from the smt-toolkit. Must be able to calculate variances
    OUTPUT:
        object that when called can return the uncertainty of a point
    """
    def un(x):
        """function that calculates the uncertainty of a metamodell at one point
        INPUT:
            x: numpy array representing a point in the designspace
        OUTPUT:
            absoulte-value of the uncertainty at one specific point
        """
        x = np.atleast_2d(x)
        
        sm_val = sm.predict_values(x)
        sm_var = sm.predict_variances(x)
        
        pyvar = sm_val+3*np.sqrt(sm_var)
        nyvar = sm_val-3*np.sqrt(sm_var)
        
        return(np.abs(pyvar - nyvar))
        
    return(un)
    
def getLowestVarianceFunction(sm):
    """defines a python object, that is able to calculate the lowest assumed variance of a metamodell
    INPUT:
        sm: a metamodell object from the smt-toolkit. Must be able to calculate variances
    OUTPUT:
        object that when called returns the lower variance of a point
    """
    def mi(x): 
        """function that calculates the lower variance of a metamodell at one point
        INPUT:
            x: numpy array representing a point in the designspace
        OUTPUT:
            lover variance of the metamodell one specific point
        """
        x = np.atleast_2d(x)
        
        sm_val = sm.predict_values(x)
        sm_var = sm.predict_variances(x)
        
        nyvar = sm_val-3*np.sqrt(sm_var)
        
        return(nyvar)
        
    return(mi)
    
def getHigestUncertainty(sm, limits):
    """calculates the 
    """
    unc_fun = getLargestUncertaintyFunction(sm)
    
    start = np.empty(limits.shape[0], dtype = float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]
    
    ret = fmin_l_bfgs_b(unc_fun, start, approx_grad=True, bounds = tuple(map(tuple,limits)))
    
    return(ret[0])
    
def getLowestVariance(sm, limits):
    min_var = getLowestVarianceFunction(sm)
    
    start = np.empty(limits.shape[0], dtype = float)
    for dim in range(limits.shape[0]):
        start[dim] = limits[dim][1] + limits[dim][0]
        
    ret = fmin_l_bfgs_b(min_var, start, approx_grad=True, bounds = tuple(map(tuple,limits)))
    
    return(ret[0])
    
