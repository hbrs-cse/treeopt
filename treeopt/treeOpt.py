import numpy as np
import os

#Import of treeopt submodules
import modules.sampling as sampling
import modules.simulate as simulate
import modules.optimize as optimize
import modules.benchmark as benchmark
import modules.metamodell as metamodell
import modules.visualize as visualize

class TreeOpt:
    """Python class, that bundles all modules for nessesary for adaptive black box metamodelling"""
    
    def __init__(self):
        """Allocates variables with standard values"""
        self.samplingMethod = sampling.latinHypercube
        self.simulateMethod = simulate.simulateBenchmarkFunction
        self.smMethod = metamodell.KRG
    
    #Functions for Data management
    def appendXData(self, xi):
        self.x = np.vstack([self.x, xi])
        
    def appendYData(self, yi):
        self.y = np.vstack([self.y, yi])
        
    #Functions to safe and load data to/from file as csv
    def writeData(self, npArray, filename):
        """Function that writes a numpy Array into a file in the threeOptData direcory
        INPUT:
            array: numpy array to be written into the file
            filename: name of the file
        """
        path = os.path.join(os.getcwd()+"/treeOptData",filename+".csv")
        np.savetxt(path, npArray, delimiter=",")
        
    def readData(self, filename):
        """Function that reads a file in the treeOptData directory and creates a numpy array with this data 
        INPUT:
            filename: name of the file to be read
        OUTPUT:
            numpy array containing the information of the file
        """
        path = os.path.join(os.getcwd()+"/treeOptData",filename+".csv")
        array = np.loadtxt(path, delimiter=",")
        return(array)
        
    #Set functions (to set functions/methods/variables prior to optimizaion)
    def setName(self, name):
        self.name = name
    
    def setSamplingMethod(self, method):
        self.samplingMethod = method
        
    def setNumDOE(self, numDOE):
        self.numDOE = numDOE
        
    def setLimits(self, limits):
        self.limits = limits
    
    def setSimulateMethod(self, method):
        self.simulateMethod = method
        
    def setBenchmarkingProblem(self, problem):
        self.benchmarkingProblem = problem
        
    def setSmMethod(self, method):
        self.smMethod = method
            
    def setAccuracyCriterion(self, method):
        self.accuracyCriterion = method
    
    #Starts one simulation run
    def simulate(self, x):
        """Function starts a Simulation and returns the resonce of the system
        
        Input: x --> Numpy-Array representing a set of parameters to be written in on simulation Input file
        Output: Numpy-Array with the system responce
        """
        if self.simulateMethod == simulate.simulateBenchmarkFunction:
            return(self.simulateMethod(self.benchmarkingProblem, x))
        else:
            print("Fehler")
        
    def startOptimization(self):
        """Function that starts the previosly parameterized adaptive optimization loop
        Input: (None directly)
        Output: Visualization of the calculated metamodell
        """
        self.x = self.samplingMethod(self.limits, self.numDOE)
        self.y = self.simulate(np.atleast_2d(self.x[0]))
        
        for xi in self.x[1:]:
            self.appendYData(self.simulate(np.atleast_2d(xi)))
        
        self.writeData(self.x, "DoeData")
        self.writeData(self.y, "DoeResponce")
        
        self.optGoal = False
        
        self.ite = 0
        self.maxite = 5
        
        while self.optGoal == False:
            
            self.sm = self.smMethod(self.x, self.y)
            
            self.nX = optimize.getLowestVariance(self.sm, self.limits)
            
            self.appendXData(self.nX)
            self.appendYData(self.simulate(np.atleast_2d(self.nX)))
            
            self.writeData(self.x, "DoeData")
            self.writeData(self.y, "DoeResponce")
            
            self.ite += 1
            
            if self.ite == self.maxite:
                self.optGoal = True
            
        vis = visualize.visualize(self)
        vis.plot()
        
