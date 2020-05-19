import numpy as np


def activation_fun(data):
    """
    Function calculates value of activation function for each element in data
    :param data: vector with weighted sum of inputs for each neuron in one layer
    :param lmbd: parameter of activation function steepness
    :return: vector with activation function values for each neuron in one layer
    """

    # result = [2/(1 + np.exp2(-lmbd*value)) - 1 for value in data]
    result = [(2/np.pi) * np.arctan(value) for value in data]
    return np.asarray(result)


def activation_fun_lin(data):
    # result = [max(min(value, 4), 0) for value in data]
    # return np.asarray(result)
    return data
    # result = [np.arctan(10*value)/np.pi + np.arctan(10*(value-1))/np.pi + np.arctan(10*(value-2))/np.pi + np.arctan(10*(value-3))/np.pi + 2 for value in data]
    # return np.asarray(result)


def activation_fun_lin_dv(data):
    # result = [1 if 0 <= value <= 4 else 0 for value in data]
    # return np.asarray(result)
    return np.ones(data.shape)
    # result = [(1/np.pi)/(1 + (10*value)**2) + (1/np.pi)/(1 + (10*(value-1))**2) + (1/np.pi)/(1 + (10*(value-2))**2) + (1/np.pi)/(1 + (10*(value-3))**2) for value in data]
    # return np.asarray(result)


def activation_fun_dv(data):
    """
    Function calculates value of derivative of activation function for each element in data
    :param data: ector with weighted sum of inputs for each neuron in one layer
    :return: vector with activation function derivative values for each neuron in one layer
    """
    result = [(2/np.pi)/(1 + value**2) for value in data]
    return np.asarray(result)
    # data_act_val = activation_fun(data, lmbd)
    # return 0.5 * (np.ones(data_act_val.shape) - np.power(data_act_val, 2))


class NeuralNetwork:

    def __init__(self, input_count, network_shape):
        """
        Constructor for the NeuralNetwork class
        :param input_count: number of inputs of the neural network
        :param network_shape: vector containing number of neurons for each layer
        """
        self.input_count = input_count
        self.network_shape = network_shape

        # Create list of matrices containing coefficients - each row is one neuron of given layer
        # end each column is for one input (previous neuron or network input) + the last one is bias
        self.neuron_layers = []
        self.layers_output = []
        self.layers_net = []
        self.layers_act_dv = []

        # Create first layer of network - first array of random coefficients
        first_layer = np.random.random([network_shape[0], input_count + 1]) * 1
        self.neuron_layers.append(first_layer)

        # Create rest of the layers with random coefficients
        for layer_count in range(1, network_shape.shape[0]):
            layer = np.random.random([network_shape[layer_count], network_shape[layer_count - 1] + 1]) * 1
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
                # x = net
                # derivative = np.ones(net.shape)

            # Add calculated outputs of one layer to the list
            self.layers_output.append(x)
            self.layers_net.append(net)
            self.layers_act_dv.append(derivative)
        return self.layers_output[-1]

    def propagate(self, nn_input, desired_output):
        layers_dq_ds = []
        layers_dq_dth = []

        self.run(nn_input)

        for layer_id in range(len(self.neuron_layers) - 1, -1, -1):
            if layer_id == len(self.neuron_layers) - 1:
                dq_dy = 2*(self.layers_output[layer_id] - desired_output)
            else:
                dq_dy = layers_dq_ds[0] @ (self.neuron_layers[layer_id + 1][:, 0:-1])

            dq_ds = dq_dy * self.layers_act_dv[layer_id]
            if layer_id > 0:
                dq_dth = np.outer(dq_ds, np.append(self.layers_output[layer_id - 1], 1))
            else:
                dq_dth = np.outer(dq_ds, np.append(nn_input, 1))

            layers_dq_ds.insert(0, dq_ds)
            layers_dq_dth.insert(0, dq_dth)

        for layer_id in range(len(self.neuron_layers)):
            self.neuron_layers[layer_id] -= 1 * layers_dq_dth[layer_id]

        return sum(pow(self.layers_output[-1] - desired_output, 2))


# xy = np.asarray([-2, -1, 0, 1, 2])
# a = activation_fun(xy, 2)
# print(activation_fun_dv(a))
nn = NeuralNetwork(2, np.asarray([10, 1]))
# inp = np.asarray([1])
# outp = np.asarray([6, 2, 4, 8, 10])
arr = np.loadtxt("tiny.txt")
print(arr)
for k in range(1000):
    suma = 0
    for row in arr:
        suma += nn.propagate(row[1:], 1 if row[0] == 1 else -1)
    for row in arr:
        # print(row[0])
        print(nn.run(row[1:]))
    print(suma)
    if suma < 0.1:
        break

print(k)
print(suma)
# for k in range(20):
#     suma = 0
#     for i in range(100):
#         for j in range(100):
#             inp = np.asarray([i, j])
#             outp = np.asarray([4 if j > i else 0, 3 if 2*j > i else 1])
#             suma += nn.propagate(inp, outp)
#     print(suma)
#     print(nn.run(np.asarray([2, 3])))


# b = np.asarray([[1, 2, 3], [4, 5, 6]])
# c = np.asarray([-1, 2])

# print(b@np.append(c, 1))
