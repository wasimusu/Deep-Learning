"""
Generating data and adding gaussian noise to it.
And applying linear regression to it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

np.random.seed(30)  # Fixing random number generator so that I get the same data on every run

# Define the parameters of the logger - filename, loggername, mode of writing, level of logging
log_filename = 'regression.csv'
logging.basicConfig(filename=log_filename, filemode='a', level=logging.CRITICAL)
logging.info(",Data, Poly_Degree, Fit, MSE, TRAIN_ERROR_PER_PREDICTION, TEST_ERROR_PER_PREDICTION, COEFFS")


def vander_matrix(X, degree=4):
    """ Generate a vandermonde matrix for fitting given degree polynomial equation """
    assert degree > 0
    assert len(X) > 0

    X = np.asarray(X)
    order = degree + 1
    v = np.empty((len(X), order))
    v[:, degree] = 1
    for i in range(0, degree):
        v[:, i] = X ** (degree - i)

    return v


def generate_data(N, sigma):
    gen_data = []  # Generated data container

    for num_points in N:
        for sigma_value in sigma:
            noise = np.random.normal(0, sigma_value, num_points)
            x = np.random.uniform(0, 3, num_points)
            y = x ** 2 - 3 * x + 1 + noise  # Compute y from x

            gen_data.append((x, y))

    return gen_data


def linear_regression(X, y, degree=2):
    XV = vander_matrix(X, degree)
    coeffs = np.linalg.inv(np.dot(XV.transpose(), XV)).dot(np.dot(XV.transpose(), y))
    fit_y = fit(X, coeffs)
    mse = np.sum((fit_y - y) ** 2)
    return coeffs, mse


def ridge_regression(X, Y, degree=2, lambda_p=0):
    """
    :param lambda_p : regularization parameter
    :param degree : degree of polynomial fit
    """
    XV = vander_matrix(X, degree)
    XVTXV = np.dot(XV.transpose(), XV)
    coeffs = np.linalg.inv(XVTXV + lambda_p * np.eye(XVTXV.shape[0])).dot(np.dot(XV.transpose(), Y))
    fit_y = fit(X, coeffs)
    mse = np.sum((fit_y - Y) ** 2)
    return coeffs, mse


def fit(X, coeff):
    """ Fit X to the coefficients computed after fitting polynomial"""
    X = vander_matrix(X, len(coeff) - 1)
    Y = np.dot(X, coeff)
    return Y


def check_fit(train_X, train_y, test_X, test_y, coeff):
    train_y_fitted = fit(train_X, coeff)
    test_y_fitted = fit(test_X, coeff)

    # The errors has been normalized to length so that lengthy vectors do not have higher errors
    train_error = np.sum((train_y - train_y_fitted) ** 2) / len(train_X)
    test_error = np.sum((test_y - test_y_fitted) ** 2) / len(test_y)

    model = "Overfit"
    if test_error < train_error:
        model = "Underfit"

    return model, train_error, test_error


def split_data(data):
    """
    Split data into test and train data for cross validation.
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
    # Get the data with the following parameters
    N = [300, 400, 600, 1200]
    sigma = [0.05]
    data = generate_data(N, sigma)

    coeffs = []
    degree = 20
    lambdas = [0, 1, 2]
    train_data, test_data = split_data(data)

    # Make a list of types of data
    properties = []
    for num_point in N:
        for sigma_value in sigma:
            properties.append("N : {} Sigma : {}".format(num_point, sigma_value))

    color = ['black', 'green', 'gray', 'red']

    for i, [X, y] in enumerate(data):
        plt.scatter(X, y, color='black', linewidths=0.01)
        patch = [
            mpatches.Patch(color=color[0], label='Original'),
            mpatches.Patch(color=color[1], label='9 Deg lambda 0'),
            mpatches.Patch(color=color[2], label='9 Deg lambda 1'),
            mpatches.Patch(color=color[3], label='9 Deg lambda 2')]
        plt.legend(handles=patch)
        plt.title('Fitting polynomial eq to {}'.format(properties[i]))

        # Fit different values of lambda
        for index, l in enumerate(lambdas):
            coeff, mse = ridge_regression(X, y, degree, lambda_p=l)

            coeffs.append(coeff)
            plt.scatter(X, fit(X, coeff), c=color[index + 1], linewidths=0.1)
            fitting, train_error, test_error = check_fit(X, y, test_data[i][0], test_data[i][1], coeff)
            summary = ",{}, {}, {}, {}, {}, {}, {}".format(properties[i], degree, fitting, np.round(mse, 3),
                                                           np.round(train_error, 3),
                                                           np.round(test_error, 3), np.round(coeff, 3))
            print(summary + "\n")
            logging.info(summary)

        plt.show()
        plt.pause(1)
