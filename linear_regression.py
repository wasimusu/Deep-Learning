"""
Generating data and adding gaussian noise to it.
And applying linear regression to it.
"""

import numpy as np

np.random.seed(3)


def generate_data():
    N = [15, 100]
    sigma = [0.01, 0.05, 0.2]

    gen_data = []

    for num_points in N:
        for sigma_value in sigma:
            noise = np.random.normal(0, sigma_value, num_points)
            x = np.random.uniform(0, 3, num_points)
            # y = x ** 2 - 3 * x + 1  # Compute y from x

            blank = np.ones((num_points, 2))
            blank[:, 1] = x
            x = blank
            w = np.array([3, 2])  # y = 2x + 3
            y = np.dot(x, w) + noise  # y = 2x + 3 + noise

            gen_data.append((x, y))
    return gen_data


def linear_regression(X, y):
    xtx_inverse = np.linalg.inv(np.dot(X.transpose(), X))
    xty = np.dot(X.transpose(), y)
    weights = xtx_inverse.dot(xty)
    print(weights)
    return weights


def linear_regression_inplace(X, y):
    weights = np.linalg.inv(np.dot(X.transpose(), X)).dot(np.dot(X.transpose(), y))
    return weights


def check_fit(train_X, train_y, test_X, test_y, weight):
    fit_train_y = np.dot(train_X, weight)
    train_error = np.sum((train_y - fit_train_y) ** 2)

    fit_test_y = np.dot(test_X, weight)
    test_error = np.sum((test_y - fit_test_y) ** 2)

    if test_error > train_error:
        print("Overfit")
    else:
        print("Underfit")


if __name__ == '__main__':
    data = generate_data()
    weights = []
    for [X, y] in data:
        weights.append(linear_regression(X, y))

    for weight, [X, y] in zip(weights, data):
        check_fit(X, y, X, y, weight)
