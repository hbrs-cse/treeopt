import numpy as np
import subprocess


def execute_python_function(fun, x, args):
    """
    Returns the function value of a benchmarking function
    :param fun: Python object representing the benchmarking function
    :type fun: Python function
    :param x: Numpy array representing a point on which the benchmarkin
    function is to be evaluated
    :type x: Numpy array
    :return: Function Value of the function at the point x
    :rtype: Numpy array
    """

    return fun(x)


def write_data(filePath, Data):
    """
    Writes the Data into a File
    :param filePath: Path of the File where Data is to be written.
    :type filePath: String
    :param Data:  Numpy array containing the Data which is to be written into
    the file.
    :type Data: Numpy array
    :return: Nothing
    :rtype: None
    """

    np.savetxt(filePath, Data)


def read_data(filePath):
    """
    Returns what is stored in the file at :filePath
    :param filePath: Path of the File to be read
    :type filePath: String
    :return: Numpy array with the content of the File
    :rtype: Numpy array
    """

    data = np.loadtxt(filePath)

    return data


def start_programm(startString):
    """
    Starts a Programm with a File to be executed
    :param programm: Name of the programm
    :type programm: TYPE
    :param simulationFile: DESCRIPTION
    :type simulationFile: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    subprocess.call([startString])


def simulate_external_programm(inputFile, outputFile, simFile, program, x):
    """
    Writes Parameters into the inputFile, starts the Simulation and returns the
    result wich got written into the outputFile
    :param inputFile:  Absolute Path of the inputFile
    :type inputFile: String
    :param outputFile: Absolute Path of the outputFile
    :type outputFile: String
    :param simFile: Absolute Path of the File storing the Simulation to be
    executed
    :type simFile: String
    :param program: Name of the programm to be executed
    :type program: String
    :param X: Parameters to be written into the input File
    :type X: Numpy array
    :return: Nothing
    :rtype: None
    """

    write_data(inputFile, x)
    print("Started Simulation")
    start_programm(program, simFile)
    y = read_data(outputFile)
    print("\nSimulation Completed")
    return y
