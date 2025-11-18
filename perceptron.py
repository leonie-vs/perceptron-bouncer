import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bias = 0.0
        self.weights = self.create_weights()
    
    def create_weights(self):
        return np.zeros(self.input_size)



