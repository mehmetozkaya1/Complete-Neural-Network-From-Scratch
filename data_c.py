# Importing necessary libraries
from keras.datasets import mnist
import numpy as np

# Load the data
(x1, y1), (x2, y2) = mnist.load_data() # Load data using Keras

# Functions that creates the data
def create_x_train_data(x, number_of_ex): # Take number of examples and the whole data
    x_train = np.array([ex.reshape(784, ) / 255 for ex in x[:number_of_ex]])
    return x_train

def create_x_test_data(x, number_of_ex):
    x_test = np.array([ex.reshape(784, ) / 255 for ex in x[:number_of_ex]])
    return x_test

def create_y_train_data(y, number_of_ex): # Take number of examples and the whole data
    y = y[:number_of_ex] # Take a piece of data
    y_train = [] # Create an empty list
    for i in range(len(y)):
        empty_list = np.zeros(10) # Create a numpy array with ten zeros
        empty_list[y[i]] = 1 # Asign one for the value
        y_train.append(empty_list) # Append it to the list
    y_train = np.array([ex.reshape(10, ) for ex in y_train]) # Reshape the data

    return y_train

def create_y_test_data(y, number_of_ex): # Take number of examples and the whole data
    y = y[:number_of_ex] # Take a piece of data
    y_test = [] # Create an empty list
    for i in range(len(y)):
        empty_list = np.zeros(10) # Create a numpy array with ten zeros
        empty_list[y[i]] = 1 # Asign one for the value
        y_test.append(empty_list) # Append it to the list
    y_test = np.array([ex.reshape(10, ) for ex in y_test]) # Reshape the data

    return y_test

x_train = create_x_train_data(x1, 3000)
y_train = create_y_train_data(y1, 3000)
x_test = create_x_test_data(x2, 3000)
y_test = create_y_test_data(y2, 3000)