import numpy as np

features = [
    "wears_i_love_javascript_tshirt",
    "is_carrying_menacing_rubber_duck",
    "tries_to_pay_with_pennies",
    "jumps_the_queue",
]

# Each line represents a person, and whether or not they have the features above
dataset = [
    [1, 1, 1, 1],  
    [1, 1, 1, 0],          
    [1, 1, 0, 0],  
	[0, 1, 1, 0],  
	[0, 0, 0, 1],  
    [0, 1, 0, 0],  
	[0, 0, 0, 0],  
]

# This list represents whether the people above are (0 = not allowed in) or (1 = allowed in)
targets = [0, 0, 0, 1, 1, 1, 1]  

def get_training_data():
    training_data = {
        'features': np.array(features),
        'dataset': np.array(dataset),
        'targets': np.array(targets)
    }
    return training_data

#print(get_training_data())
