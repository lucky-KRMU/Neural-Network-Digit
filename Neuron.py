import math
import numpy as np

class Neuron:
    '''
    This is a Sigmoid neuron class
    '''
    
    
    
    def __init__(self, inputs: list):
        '''
        This is the constructor of the class it would define the weights, biases and
        would also flatten the input multidimensional array using numpy
        '''
    
        self.inputs = np.array(inputs).flatten() # flattening the array
        self.param_num = len(self.inputs)   # calculating the total number of parameters
        self.weights = np.zeros(self.param_num) # initializing the initial weights
        self.bias = 0 # initializing the bias initially
        