import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.bias = 0.0
        self.weights = self.create_weights()
    
    def create_weights(self):
        return np.zeros(self.input_size)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict_probability(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return probabilities
    
    def predict(self, X):
        probabilites = self.predict_probability(X)
        result = []
        for i in probabilites:
            if i >= 0.5:
                result.append(1)
            else:
                result.append(0)
        return result
    
    def train(self, X, y, epochs):
        for i in range(epochs):
            predictions = self.predict_probability(X)
            error = predictions - y
            loss = np.mean(error**2)
            grad = error * predictions * (1 - predictions)
            self.weights -= self.learning_rate * np.dot(X.T, grad)
            self.bias -= self.learning_rate * np.sum(grad)
        

    




