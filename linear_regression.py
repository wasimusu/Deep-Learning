"""
Generating data and adding gaussian noise to it.
And applying linear regression to it.
"""

import numpy as np

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


def linear_regression(X, y, degree=9):
    X = vander_matrix(X, degree)
    xtx_inverse = np.linalg.inv(np.dot(X.transpose(), X))
    xty = np.dot(X.transpose(), y)
    weights = xtx_inverse.dot(xty)

    print(np.round(weights, 3))
    return weights


def linear_regression_inplace(X, y, degree=2):
    X = vander_matrix(X, degree)
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
        # weights.append(linear_regression_inplace(X, y))
        # break

    # for weight, [X, y] in zip(weights, data):
    #     check_fit(X, y, X, y, weight)
