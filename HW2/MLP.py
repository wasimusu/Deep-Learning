""" Implementing Multilayer Perceptron with Sigmoid and Thresholding """
import numpy as np


def threshold(x):
    """ Apply thresholding for perceptron """
    return 1 if x > 0 else 0


def sigmoid(x):
    """ Compute the sigmoid value of wx + b """
    return 1 / (1 + np.exp(-x))


class Perceptron():
    def __init__(self, activation_type='threshold'):
        """ Define the weights of the network """

        # Weight of input / first layer of perceptron
        self.w_I = np.array([[0.6, -0.7, 0.5, -0.4],
                             [0.4, -0.6, 0.8, -0.5]])

        # Weight of output layer of perceptron
        self.w_O = np.array([1, 1, -0.4])

        self.activation_map = {'threshold': threshold, 'sigmoid': sigmoid}
        self.activation_func = self.activation_map[activation_type]

    def forward(self, x):
        """ Propagate forward the input x """

        x = np.dot(x, self.w_I.transpose())
        x = np.append(x, 1)
        x = np.dot(x, self.w_O.transpose())
        x = self.activation_func(x)
        return x


if __name__ == '__main__':
    mlp = Perceptron(activation_type='sigmoid')
    # mlp = Perceptron(activation_type='threshold')
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = np.array([i, j, k, 1]).reshape(1, -1)
                output = mlp.forward(x)
                print(x, np.round(output))
