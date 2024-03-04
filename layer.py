import numpy as np # Importing necessary library

class Layer: # Layer class which include the neurons
    def __init__(self, units, activation, input_size):
        self.input = None # Layer's input   
        self.output = None # Layer's output
        self.units = units # Number of neurons in the layer
        self.activation = activation # Activation function of the layer
        self.input_size = input_size # Input size of the layer
        self.weights = np.random.randn(self.units, self.input_size) # Layer's weights
        self.bias = np.random.randn(self.units, 1) # Layer's bias
        self.set_weights_init() # Set initial weights by looking at the activation function

    def set_weights_init(self): # Setting the best initial weights by looking at the activation function
        activation = self.activation # Activation
        if activation == "relu": # If the activation is relu (He initialization applied)
            variance = 2.0 / self.input_size # Variance
            std_dev = np.sqrt(variance) # Standard deviation
            self.weights = np.random.normal(0, std_dev, size=(self.units, self.input_size)) # Setting the weights
        elif activation == "sigmoid" or activation == "tanh": # If the activation is sigmoid or tanh (Xavier initialization applied) 
            variance = 2.0 / (self.input_size + self.units) # Variance
            std_dev = np.sqrt(variance) # Standard deviation
            self.weights = np.random.normal(0, std_dev, size=(self.units, self.input_size)) # Setting the weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self): # get_weights method which returns the weights of the layer
        return self.weights, self.bias

    def activation_func(self, z): # Layer's activation function
        activation = self.activation # Activation function
        if activation == "relu": # If it is relu
            return np.maximum(0, z)
        elif activation == "sigmoid": # If it is sigmoid
            return 1 / (1 + np.exp(-z))
        elif activation == "tanh": # If it is tanh
            return np.tanh(z)
        elif activation == "linear": # If it is linear
            return z
        elif activation == "softmax": # If it is softmax
            exp_z = np.exp(z)
            sum = exp_z.sum()
            softmax_z = np.round(exp_z/sum,3)
            return softmax_z

    def activation_func_prime(self, z): # The derivative of the activation function
        activation = self.activation # Activation
        if activation == "relu": # relu's derivative
            return z > 0
        elif activation == "sigmoid": # sigmoid's derivative
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif activation == "tanh": # tanh's derivative
            tanh = np.tanh(z)
            return 1 - tanh ** 2
        elif activation == "linear": # linear's derivative
            return 1
        elif activation == "softmax": # softmax's derivative
            exp_z = np.exp(z)
            sum = exp_z.sum()
            softmax_z = np.round(exp_z/sum, 3)
            return softmax_z * (1 - softmax_z)

    def forward(self, input): # Forward function which creates the output of the layer
        self.input = input # Set the input
        self.output = self.activation_func(np.dot(self.weights, self.input) + self.bias) # Create the output and set it
        return self.output # Return the output
    
    def backward(self, output_gradient, learning_rate): # Backward function which takes the next neuron's backward input and computes the gradient of thw weights and bias and returns the previous neurons backward input
        output_gradient = np.multiply(output_gradient, self.activation_func_prime(self.output)) # The derivative of the activation function with respect to the output of the layer

        dj_dw = np.dot(output_gradient, self.input.T) # The gradient of the weights 
        dj_db = output_gradient # The gradient of the bias
        dj_dx = np.dot(self.weights.T, output_gradient) # The backward input of the previous neuron

        self.weights -= learning_rate * dj_dw # Gradient descent of the weights
        self.bias -= learning_rate * dj_db # Gradient descent of the bias

        return dj_dx # Return the backward input of the previos neuron