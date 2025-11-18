Building a Perceptron in Python 

- Requirements for this repo can be found in **requirements.txt**.
- The Perceptron class is defined in **perceptron.py**, and a perceptron model can be built by creating an instance of this class with a chosen input_size passed as argument.
- The unit tests for the Perceptron class can be found in **test_perceptron.py** and can be executed using pytest. 
- The training data is in **data.py**, and the function get_training_data returns a dictionary with the features, dataset, and targets as Numpy arrays. 
- A perceptron model can then be trained using the train method (example usage in **main.py**).
- The code in **cli_playground.py** creates a CLI which allows users to enter data about customers. When running this file, users can answer the four questions with 0 or 1, which will result in telling them whether or not they're welcome.