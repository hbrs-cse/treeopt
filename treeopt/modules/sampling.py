import smt.sampling_methods as smtSam

def latinHypercube(limits, ndoe):
    """Function to approach a designspace by using a latin hypercube algorithm
    INPUT:
        limits: numpy array representing the limit of the desingnspace
        ndoe: number of samplingpoints to be generated in the designspace
    OUTPUT:
        Numpy-array containg the sampled points
    """
    sampling = smtSam.LHS(xlimits = limits, criterion='m')
    x = sampling(ndoe)
    return(x)
    
def fullFaktorial(limits, ndoe):
    """Function to approach a designspace by using full factorial grid
    INPUT:
        limits: numpy array representing the limit of the desingnspace
        ndoe: number of samplingpoints to be generated in the designspace
    OUTPUT:
        Numpy-array containg the sampled points
    """
    sampling = smtSam.FullFactorial(xlimits = limits)
    x = sampling(ndoe)
    return(x)
    
def Random(limits, ndoe):
    """Function to approach a designspace randomly picking points in the designspace
    INPUT:
        limits: numpy array representing the limit of the desingnspace
        ndoe: number of samplingpoints to be generated in the designspace
    OUTPUT:
        Numpy-array containg the sampled points
    """
    sampling = smtSam.Random(xlimits = limits)
    x = sampling(ndoe)
    return(x)        