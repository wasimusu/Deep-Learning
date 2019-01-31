""" Implementing Multilayer Perceptron with two activation functions : Sigmoid and Simple Thresholding """
import numpy as np


def threshold(x):
    """ Simple thresholding function with threshold at 0. """
    return 1 if x > 0 else 0


def sigmoid(x):
    """ Compute the sigmoid value of wx + b """
    return 1 / (1 + np.exp(-x))


class Perceptron():
    def __init__(self, activation='threshold'):
        """
        Define the weights of the network.
        The weights and biases are written together as weights.
        """

        # Weight of input / first layer of perceptron
        self.w_I = np.array([[0.6, -0.7, 0.5, -0.4],
                             [0.4, -0.6, 0.8, -0.5]])

        # Weight of output layer of perceptron
        self.w_O = np.array([1, 1, -0.4])

        # Dictionary to let user chose their activation type
        self.activation_map = {'threshold': threshold, 'sigmoid': sigmoid}

        # Activation function chosen by the user
        self.activation = self.activation_map[activation]
        self.activation = np.vectorize(self.activation)  # So that activation functions work for numpy arrays

    def forward(self, x):
        """ Propagate forward the input x """

        # Propagate through first layer and activate
        x = np.dot(x, self.w_I.transpose())
        x = self.activation(x)

        x = np.append(x, 1)  # Append 1 for the bias

        # Propagate through second layer and activate
        x = np.dot(x, self.w_O.transpose())
        x = self.activation(x)
        return x


if __name__ == '__main__':
    activations = ['threshold', 'sigmoid']
    for activation_type in activations:
        mlp = Perceptron(activation=activation_type)

        # Generate the binary number from 000 to 111
        print("Activation type : ", activation_type)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = np.array([i, j, k, 1]).reshape(1, -1)
                    output = mlp.forward(x)
                    print(x, np.round(output, 2))
