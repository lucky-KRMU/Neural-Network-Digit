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
        
    def weighted_sum_func(self) -> float:
        '''
        This is the method to claculate the weighted sum
        -> this is not the activation
        '''
        
        sum = 0
        self.weighted_sum = 0
        for x,y in zip(self.weights, self.inputs):
            sum += x * y
        
        self.weighted_sum = sum + self.bias
        return self.weighted_sum
        