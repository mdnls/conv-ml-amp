import importlib
import numpy
import scipy
import os

'''
Note from Authors (NeurIPS 2022 submit): we have removed the functionality below providing GPU support for our code. Currently, this is buggy, and we did not use it to generate the figures in the attached text. 
'''

def GET_NUMERICAL_LIB(device="default"):
    no_cupy = (os.environ.get('MLAMP_NO_CUPY') is not None)
    #if(((not no_cupy) and device == "default" and (importlib.util.find_spec("cupy") is not None)) or (device == "gpu")):
    if(False):
        import cupy
        return cupy
    else:
        return numpy

def GET_DEVICE(device="default"):
    no_cupy = (os.environ.get('MLAMP_NO_CUPY') is not None)
    #if(((not no_cupy) and device == "default" and (importlib.util.find_spec("cupy") is not None)) or (device == "gpu")):
    if(False):
        return "gpu"
    else:
        return "cpu"

def to_cpu(x):
    if(isinstance(x, numpy.ndarray) or isinstance(x+0.0, float)):
        return x
    else:
        return numpy.array(x.get())