import numpy as np
from perceptron import Perceptron

def test_perceptron_init():
	p = Perceptron(input_size=3)
	assert isinstance(p.weights, np.ndarray)
	assert p.weights.shape == (3,)
	assert np.all(p.weights == 0.0) 
	assert p.learning_rate == 0.1 
	assert p.bias == 0.0

def test_perceptron_sigmoid():
	p = Perceptron(input_size=3)
	assert np.isclose(p.sigmoid(0), 0.5), "Sigmoid(0) should be 0.5"
	assert np.isclose(p.sigmoid(100), 1.0, atol=1e-6), "Sigmoid(100) should approach 1"
	assert np.isclose(p.sigmoid(-100), 0.0, atol=1e-6), "Sigmoid(-100) should approach 0"
	input_array = np.array([0, 1, -1])
	output = p.sigmoid(input_array)
	expected = 1 / (1 + np.exp(-input_array))
	assert np.allclose(output, expected), "Sigmoid should handle vector input correctly"
	
