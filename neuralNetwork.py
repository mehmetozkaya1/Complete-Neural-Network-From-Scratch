# Importing necessary libraries
import numpy as np
import math

### Optimizer (Ex. : Adam) should be added
class NeuralNetwork: # NeuralNetwork class which includes the layers and organises the forward and backward propogation
    def __init__(self, layers, x_train, y_train, cost_func):
        self.layers = layers # Layers of the neural network
        self.x_train, self.y_train = np.reshape(x_train, (len(x_train), len(x_train[0]), 1)), np.reshape(y_train, (len(y_train), len(y_train[0]), 1)) # Train datas
        self.cost_func = cost_func # The cost function of the whole network

        # self.x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4,2,1))
        # self.y_train = np.reshape([[0], [1], [1], [0]], (4,1,1))

    def forward_pass(self, x): # forward_pass method which operates the forward propogation between the layers
        output = x # Initial input
        for layer in self.layers: # For each layer
            output = layer.forward(output) # Compute the output using the previous layer's output
        return output # And return the final output

    def backward_pass(self, grad, learning_rate): # backward_pass method which operates the forward propogation between the layers. Take the gradient of the activation function with respect to the output of the output layer
        for layer in reversed(self.layers): # For each layer (reversed)
            grad = layer.backward(grad, learning_rate) # Set new weights and bias and return the backward input of the previous layer.

    def compute_cost(self, output, y): # compute_cost method which computes the cost
        cost_func = self.cost_func
        if cost_func == "MSE": # If the cost function is mean squared error
            cost = 0. # Initial cost value
            cost = np.mean(np.power(y - output, 2)) # Calculate the cost
        return cost # Return the cost

    def train(self, epochs, learning_rate): # train method which trains the model using the neural network we created
        cost_his = [] # Empty list to hold the costs
        for epoch in range(epochs): # For each epoch
            cost = 0. # Initial cost value
            for x, y in zip(self.x_train, self.y_train): # For each x and y in the train set
                output = self.forward_pass(x) # Compute the final output
                cost += self.compute_cost(output, y) # Compute the cost (compute the cost for each example in the training set and add it to the cost)
                grad = (output - y) # Compute the gradient using the final output and the actual value

                self.backward_pass(grad, learning_rate) # Operate backpropogation

            cost /= len(x) # Find the final cost
            cost_his.append(cost) # Append the cost to the list
            if epoch % math.ceil(epochs / 10) == 0:
                print(f"Epoch {epoch:4d}: Cost {cost_his[-1]:8.10f}") # Print the cost for each epoch
        return cost_his # Return the cost_history

    def predict(self, input_x): # Take the input
        outputs = [] # Empty list which holds the output of the neural network
        for x in np.reshape(np.array(input_x) ,(len(input_x),784,1)): # For each input in the input_x
            output = x # Initial output
            output = self.forward_pass(output) # Compute the output of the neural network
            outputs.append(output) # Append the value
        ### Layer's activation funct'a göre output oluştur. ###
        return np.array([output.reshape(10,) for output in outputs]) # Return the outputs

    def score(self, y_pred, y_test): # score method which returns the accuracy of the neural network
        print("Predicted numbers (Max probabilities) : \n", y_pred)
        print("Expected numbers : \n", y_test)
        counter = 0. # Counter
        for i in range(len(y_pred)): 
            if y_pred[i] == y_test[i]: # If the matching indexes are equal
                counter += 1.0 # Add 1 to the counter
        score = counter / float(len(y_pred)) # Calculate the score
        return score # Return the accuracy
    
    def save_neural_network(self, file_path):
        weights_dict = {}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            weights_dict[f"Layer {i + 1}"] = layer.weights
        return weights_dict

    def load_neural_network(self, file_path):
        pass

    def __str__(self):
        pass