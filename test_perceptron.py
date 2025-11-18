import numpy as np
from perceptron import Perceptron

def test_perceptron_init():
	p = Perceptron(input_size=3)
	assert isinstance(p.weights, np.ndarray)
	assert p.weights.shape == (3,)
	assert np.all(p.weights == 0.0) 
	assert p.learning_rate == 0.1 
	assert p.bias == 0.0