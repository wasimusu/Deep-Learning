""" Implementing Multilayer Perceptron with two activation functions : Sigmoid and Simple steping """
import numpy as np


def step(x):
    """ Simple steping function with step at 0. """
    return 1 if x > 0 else 0


def sigmoid(x):
    """ Compute the sigmoid value of wx + b """
    return 1 / (1 + np.exp(-x))


class Perceptron():
    def __init__(self, activation='step'):
        """
        Define the weights of the network.
        The weights and biases are written together as weights.
        """

        # Weight of input / first layer of perceptron
        self.w_IH = np.array([[0.15, 0.20, 1],
                              [0.25, 0.30, 1]])

        # Weight of input / first layer of perceptron
        self.w_HO = np.array([[0.4, 0.45, 1],
                              [0.50, 0.55, 1]])

        # Dictionary to let user chose their activation type
        self.activation_map = {'step': step, 'sigmoid': sigmoid}

        # Activation function chosen by the user
        self.activation = self.activation_map[activation]
        self.activation = np.vectorize(self.activation)  # So that activation functions work for numpy arrays

    def forward(self, x):
        """ Propagate forward the input x """

        # Propagate through first layer and activate
        x = np.dot(x, self.w_IH.transpose())
        x = self.activation(x)
        print(x)

        x = np.append(x, 0.60)  # Append 0.60 for the bias

        # Propagate through second layer and activate
        x = np.dot(x, self.w_HO.transpose())
        x = self.activation(x)
        print(x)

        return x

    def loss(self, output, target):
        """
        Compute cross entropy cost function
        """
        import math

        loss = 0
        for y, a in zip(target, output):
            node_loss = -1 * (y * math.log(a, math.e) + (1 - y) * math.log(1 - a, math.e)) / len(output)
            print(node_loss)
            loss += node_loss

        # loss = F.cross_entropy(x, target)
        # loss = np.sum((x - target) ** 2) / 2
        return loss


if __name__ == '__main__':
    activations = ['sigmoid']

    target = np.array([0.01, 0.99])

    mlp = Perceptron(activation=activations[0])
    output = mlp.forward([0.05, 0.10, .35])
    loss = mlp.loss(output, target)
    print("Output : ", output)
    print("Loss : ", loss)
