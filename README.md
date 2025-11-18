<h1 style="color:Teal;">Perceptron Bouncer</h1> 

<p>
In this repo I built a Perceptron-based neural network which can identify whether or not a customer is allowed to enter a pub based on these four features:</p>

<ul style="color:Teal;">
    <li>Are they wearing an 'I love JavaScript' T-shirt?</li>
    <li>Are they carrying a menacing rubber duck?</li>
    <li>Are they trying to pay with pennies?</li>
    <li>Are they jumpping the queue?</li>
</ul>


<p>
Requirements for this repo can be found in **requirements.txt**.

The Perceptron class is defined in <span style="color:Teal;">**perceptron.py**</span>, and a perceptron model can be built by creating an instance of this class with a chosen input_size passed as argument.

The unit tests for the Perceptron class can be found in <span style="color:Teal;">**test_perceptron.py**</span> and can be executed using pytest. 

The training data is in <span style="color:Teal;">**data.py**</span>, and the function get_training_data returns a dictionary with the features, dataset, and targets as Numpy arrays. 

A perceptron model can then be trained using the train method (example usage in <span style="color:Teal;">**main.py**</span>).

The code in <span style="color:Teal;">**cli_playground.py**</span> creates a CLI which allows users to enter data about customers. When running this file, users can answer the four questions with 0 or 1, which will result in telling them whether or not they're welcome.</p>