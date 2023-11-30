import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def symbolic_interval_analysis(network, input_interval):
    numLayer = len(network.layers)  # Assuming the network has a 'layers' attribute
    layerSizes = [layer.size for layer in network.layers]  # Assuming each layer has a 'size' attribute

    # Initialize symbolic equation eq = (equp, eqlow)
    eq = (np.zeros(layerSizes[0]), np.zeros(layerSizes[0]))  # Assuming input size is the same as the first layer size

    # Cache mask matrix needed in backward propagation
    R = np.zeros((numLayer, max(layerSizes)))

    for layer in range(1, numLayer + 1):
        # Matmul equations with weights as interval
        weights = network.layers[layer - 1].weights  # Assuming each layer has 'weights' attribute
        eq = np.matmul(weights, eq)

        # Update the output ranges for each node
        if layer != numLayer:
            for i in range(layerSizes[layer - 1]):
                if eq[0][i] <= 0:
                    # Update to 0
                    R[layer - 1][i] = 0
                    eq[0][i] = eq[1][i] = 0
                elif eq[1][i] >= 0:
                    # Keep dependency
                    R[layer - 1][i] = 1
                else:
                    # Concretization
                    R[layer - 1][i] = 0.5
                    eq[1][i] = 0
                    if eq[0][i] > 0:
                        eq[0][i] = eq[0][i]

    # Output the results
    output = (np.min(eq), np.max(eq))
    return R, output

# Example usage (network and input_interval need to be defined as per specific requirements)
# network = ...  # Define or load your neural network here
# input_interval = ...  # Define your input interval here
# R, output = symbolic_interval_analysis(network, input_interval)
# print("Output interval:", output)


import numpy as np

class Layer:
    def __init__(self, size, weights):
        self.size = size
        self.weights = weights

class SimpleNeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

# Example Neural Network
# Two layers with arbitrary weights for demonstration

# Layer 1 (3 neurons)
layer1 = Layer(3, np.array([[0.2, -0.5, 0.3],
                            [0.4, 0.1, -0.2],
                            [-0.3, 0.4, 0.5]]))

# Layer 2 (2 neurons)
layer2 = Layer(2, np.array([[0.3, -0.1, 0.2],
                            [-0.2, 0.2, -0.3]]))

# Create the network
network = SimpleNeuralNetwork([layer1, layer2])

# Example Input Interval
# Assuming the network takes 3 input features
input_interval = (np.array([-1, 0, 1]), np.array([1, 2, 3]))  # Lower and upper bounds

# Now you can call the function with these inputs
R, output = symbolic_interval_analysis(network, input_interval)
print("Output interval:", output)
