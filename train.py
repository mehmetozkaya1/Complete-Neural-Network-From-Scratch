# Implementing necessary libraries
import numpy as np
from neuralNetwork import NeuralNetwork
from layer import Layer
from data_c import x_train, y_train, x_test, y_test
from PIL import Image
import matplotlib.pyplot as plt

# Layers
layer1 = Layer(units=100, activation="relu", input_size=784)
layer2 = Layer(units=50, activation="relu", input_size=100)
layer3 = Layer(units=10, activation="sigmoid", input_size=50)
layers = [layer1, layer2, layer3]

# Neural network
epochs = 25
learning_rate = 0.01
cost_func = "MSE"

nn = NeuralNetwork(layers, x_train, y_train, cost_func)
cost_his = nn.train(epochs, learning_rate)
print()

# Predict
number_of_ex = 1000
y_pred = nn.predict(x_test[:number_of_ex]) # Probabilities for each digit (y_pred)
print()
numbers = [np.argmax(pred) for pred in y_pred] # Take the maximum probabilities in the lists (Predicted numbers)
answers = [np.reshape(answer, (10,)).tolist() for answer in y_test[:number_of_ex]] # Actual numbers
answers = [np.argmax(answer) for answer in answers] # Take the biggest one (Actual numbers)
print()

# Model's accuracy
score= nn.score(numbers, answers) # Accuracy of the model
print(f"Model's accuracy: %{score * 100}") # Print the score

# Visualize the cost
def show_cost_hist(cost_his):
    epochs = range(1, len(cost_his) + 1) # Ranging the epochs
    plt.plot(epochs, cost_his, c="red") # Plot the cost history 
    plt.title('Cost Over Epochs') # Set the title
    plt.xlabel('Epoch') # Set x label's name
    plt.ylabel('Cost') # Set y label's name
    plt.show() # Show the graph

show_cost_hist(cost_his)

# Show the images
images = [image.reshape(28,28) for image in x_test[:number_of_ex]]

def show_images(images):
    for image in images:
        image = Image.fromarray((image * 1000).astype(np.uint8))

        plt.imshow(image)
        plt.show()

# show_images(images)