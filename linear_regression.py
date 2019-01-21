"""
Generating data and adding gaussian noise to it.
And applying linear regression to it.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)


def vander_matrix(X, degree=4):
    assert degree > 0
    assert len(X) > 0

    X = np.asarray(X)
    order = degree + 1
    v = np.empty((len(X), order))
    v[:, degree] = 1
    for i in range(0, degree):
        v[:, i] = X ** (degree - i)
    return v


def generate_data():
    N = [15, 100]
    sigma = [0.01, 0.05, 0.2]

    gen_data = []

    for num_points in N:
        for sigma_value in sigma:
            noise = np.random.normal(0, sigma_value, num_points)
            x = np.random.uniform(0, 3, num_points)
            y = x ** 2 - 3 * x + 1 + noise  # Compute y from x

            gen_data.append((x, y))
    return gen_data


def linear_regression(X, y, degree=2):
    X = vander_matrix(X, degree)
    weights = np.linalg.inv(np.dot(X.transpose(), X)).dot(np.dot(X.transpose(), y))
    return weights


def fit(X, weight):
    X = vander_matrix(X, len(weight) - 1)
    Y = np.dot(X, weight)
    return Y


def check_fit(train_X, train_y, test_X, test_y, weight):
    fit_train_y = fit(train_X, weight)
    fit_test_y = fit(test_X, weight)

    # The errors has been normalized to length so that lengthy vectors do not have higher errors
    train_error = np.sum((train_y - fit_train_y) ** 2) / len(train_X)
    test_error = np.sum((test_y - fit_test_y) ** 2) / len(test_Y)

    if test_error > train_error:
        print("Overfit")
    else:
        print("Underfit")


def test_and_train_data(data):
    """ Prepare data for cross validation.
     Train Data     | Test Data
     Set 1          | Set 2, 3, 4, 5, 6
     Set 2          | Set 1, 3, 4, 5, 6
     .................
     """
    testXY = []
    trainXY = data
    for i in range(len(data)):
        test_X = []
        test_Y = []
        for j in range(len(data)):
            if i != j:
                test_X += list(data[j][0])
                test_Y += list(data[j][1])
        assert len(test_X) == len(test_Y)
        testXY.append((np.asarray(test_X), np.asarray(test_Y)))
    return trainXY, testXY


if __name__ == '__main__':
    import matplotlib.patches as mpatches

    data = generate_data()
    weights = []
    degrees = [1, 2, 9]

    train_data, test_data = test_and_train_data(data)

    color = ['black', 'red', 'green', 'orange']

    for i, [X, y] in enumerate(data):

        plt.scatter(X, y, color='black')

        for index, degree in enumerate(degrees):
            weight = linear_regression(X, y, degree)
            weights.append(weight)
            plt.scatter(X, fit(X, weight), c=color[index + 1], label='1')

        plt.title('Fitting 1, 2 and 9 polynomial ')

        patch = [
            mpatches.Patch(color='black', label='Original'),
            mpatches.Patch(color='red', label='1 Deg'),
            mpatches.Patch(color='green', label='2 Deg'),
            mpatches.Patch(color='orange', label='9 Deg')]

        plt.legend(handles=patch)

        plt.show()
        plt.pause(1)

    for weight, [train_X, train_Y], [test_X, test_Y] in zip(weights, train_data, test_data):
        check_fit(train_X, train_Y, test_X, test_Y, weight)
