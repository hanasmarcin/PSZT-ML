# PSZT-ML Neural Network project
# Hanas Marcin, Tuzimek Radosław
# PW
import numpy as np


def activation_fun(data):
    """
    Function calculates value of activation function for each element in data
    :param data: vector with weighted sum of inputs for each neuron in one layer
    :param lmbd: parameter of activation function steepness
    :return: vector with activation function values for each neuron in one layer
    """
    result = [2/np.pi*np.arctan(value) for value in data]
    return np.asarray(result)


def activation_fun_dv(data):
    """
    Function calculates value of derivative of activation function for each element in data
    :param data: ector with weighted sum of inputs for each neuron in one layer
    :return: vector with activation function derivative values for each neuron in one layer
    """
    result = [2/np.pi/(1 + value**2) for value in data]
    return np.asarray(result)


class NeuralNetwork:
    """
    Class implements neural network
    """

    def __init__(self, input_count, network_shape, learning_rate):
        """
        Constructor for the NeuralNetwork class
        :param input_count: number of inputs of the neural network
        :param network_shape: vector containing number of neurons for each layer
        """
        self.learning_rate = learning_rate
        self.input_count = input_count
        self.network_shape = network_shape

        # Create list of matrices containing coefficients - each row is one neuron of given layer
        # end each column is for one input (previous neuron or network input) + the last one is bias
        self.neuron_layers = []
        self.layers_output = []
        self.layers_net = []
        self.layers_act_dv = []

        # Create first layer of network - first array of random coefficients
        np.random.seed(10)
        first_layer = (np.random.rand(network_shape[0], input_count + 1) - 0.5*np.ones([network_shape[0], input_count + 1])) / np.sqrt(input_count + 1)
        self.neuron_layers.append(first_layer)

        # Create rest of the layers with random coefficients
        for layer_count in range(1, network_shape.shape[0]):
            layer = (np.random.rand(network_shape[layer_count], network_shape[layer_count - 1] + 1) - 0.5*np.ones([network_shape[layer_count], network_shape[layer_count - 1] + 1])) / np.sqrt(network_shape[layer_count - 1] + 1)
            self.neuron_layers.append(layer)
            # print(layer_count)
        # print(self.neuron_layers)

    def run(self, x):
        """
        Method calculates network's output for given input
        :param x: vector with input for the neural network
        :return: vector with output of the neural network
        """
        assert x.shape[0] == self.input_count
        self.layers_output.clear()
        self.layers_net.clear()
        self.layers_act_dv.clear()
        for layer_id in range(len(self.neuron_layers)):
            # Calculate net values for each neuron in one layer
            net = self.neuron_layers[layer_id] @ np.append(x, 1)

            # If this is the hidden layer, calculate activation function for the net value
            if layer_id < len(self.neuron_layers) - 1:
                x = activation_fun(net)
                derivative = activation_fun_dv(net)
            else:
                x = activation_fun(net)
                derivative = activation_fun_dv(net)

            # Add calculated outputs of one layer to the list
            self.layers_output.append(x)
            self.layers_net.append(net)
            self.layers_act_dv.append(derivative)
        return self.layers_output[-1]

    def propagate(self, nn_input, desired_output):
        """
        Method changes network's weights using backpropagation method for given element
        :param nn_input: vector with input values for backpropagation
        :param desired_output: vector with desired output values for given input
        :return: value of total error for given element
        """
        layers_dq_ds = []
        layers_dq_dth = []

        self.run(nn_input)

        # For each layer, starting with output layer
        for layer_id in range(len(self.neuron_layers) - 1, -1, -1):
            if layer_id == len(self.neuron_layers) - 1:
                # If this is an output layer
                dq_dy = 2*(self.layers_output[layer_id] - desired_output)
            else:
                dq_dy = layers_dq_ds[0] @ (self.neuron_layers[layer_id + 1][:, 0:-1])

            dq_ds = dq_dy * self.layers_act_dv[layer_id]
            if layer_id > 0:
                dq_dth = np.outer(dq_ds, np.append(self.layers_output[layer_id - 1], 1))
            else:
                # If this is the first layer
                dq_dth = np.outer(dq_ds, np.append(nn_input, 1))

            # Add calculated values to lists
            layers_dq_ds.insert(0, dq_ds)
            layers_dq_dth.insert(0, dq_dth)

        # Change weights for each neuron layer
        for layer_id in range(len(self.neuron_layers)):
            self.neuron_layers[layer_id] -= self.learning_rate * layers_dq_dth[layer_id]

        return sum(pow(self.layers_output[-1] - desired_output, 2))
