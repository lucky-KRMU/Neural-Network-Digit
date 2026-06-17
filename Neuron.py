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
        self.trained = False # To check whether the training is completed or not
        self.asked_inputs = np.array(inputs).flatten()
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
    
    def train(self, traning_input: list, lr: float = 0.01, epocs: int = 10):
        '''
        This is the method to train the sigmoid neuron.
        It is taking an array of training data, which would have 4X4 pixel or 4x4 matrix that would contain 
        0s or 1s to depict or make an approximate 4.
        '''
        
        for i, o in traning_input:
            self.inputs = np.array(i).flatten()
            
            for j in range(epocs):
                # Applying the gradient descent
                
                prediction = self.activation()
                
                # cost = 0.5 * (expected_output - self.inputs[k]) ** 2
                
                # l = ∂z/∂w
                # adjustment = lr * prediction * (1 - prediction) * (prediction - self.inputs[j]) *  (self.asked_inputs[j])
                # The above statement is mathematically saying that i am trying to compare or find out the cost by 
                # comparing prediction of the entire image with just one pixel.
                
                adjustment = lr * prediction * (1 - prediction) * (prediction - o) *  (self.inputs[j])
                # if cost < .1:
                #     self.weights[k] += adjustment
                #     self.bias += lr * adjustment
                self.weights[j] += adjustment
                self.bias += lr * adjustment
        self.trained = True
                    
    def activation(self) -> float:
        '''
        This is the function to make calculate and give the output of final activation.
        '''
        
        if self.trained:
            self.inputs = self.asked_inputs
        
        
        
        self.weighted_sum_func()
        self.z = 1/(1 + math.exp((-self.weighted_sum)))
        return self.z

training_data=[
    # Changed the training data format to tuple to tell the neuron whether it's a 4 or not
    ([
        [1,0,0,1],
        [1,0,0,1],
        [1,1,1,1],
        [0,0,0,1]
    ], 1),
    ([
        [0,0,0,1],
        [0,0,1,1],
        [1,1,1,1],
        [0,0,0,1]
    ], 1),
    ([
        [0,0,1,1],
        [0,1,1,1],
        [1,1,1,1],
        [0,0,0,1]
    ], 1)
]
int_list = [
    [1,0,0,1],
    [1,0,0,1],
    [1,1,1,1],
    [0,0,0,1]
]

N = Neuron(int_list)
print(N.activation())
N.train(training_data)
print('weights: ',N.weights, '\nbias: ', N.bias)
print(N.activation())