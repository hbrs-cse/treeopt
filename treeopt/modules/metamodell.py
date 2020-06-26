import smt.surrogate_models as smt

def RBF(xt, yt):
    """Function, that trains a metamodell based on the radial basis function algorithm
    INPUT:
        xt: Array of points in which the system responce is known
        yt: Array contining the system responce
    OUTPUT:
        python object containing the trained metamodell
    """
    sm = smt.RBF(print_prediction = False, poly_degree = 0, print_global = False)
    sm.set_training_values(xt, yt)
    sm.train()
        
    return(sm)

def KRG(xt, yt):
    """Function, that trains a metamodell based on the kriging algorithm
    INPUT:
        xt: Array of points in which the system responce is known
        yt: Array contining the system responce
    OUTPUT:
        python object containing the trained metamodell
    """
    sm = smt.KRG(theta0=[1e-2])
    sm.set_training_values(xt, yt)
    sm.train()
    
    return(sm)

def IDW(xt, yt):
    sm = smt.IDW(p = 2)
    sm.set_training_values(xt,yt)
    sm.train()
    
    return(sm)