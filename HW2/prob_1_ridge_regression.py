"""
Generating data and adding gaussian noise to it.
And applying ridge regression to it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

np.random.seed(30)  # Fixing random number generator so that I get the same data on every run

# Define the parameters of the logger - filename, loggername, mode of writing, level of logging
log_filename = 'regression.csv'
logging.basicConfig(filename=log_filename, filemode='a', level=logging.INFO)
logging.info(",Data, Poly_Degree, lambda, Fit, MSE, TRAIN_ERROR_PER_PREDICTION, TEST_ERROR_PER_PREDICTION, COEFFS")


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
    """ Generate data with given number of points N and sigma """
    noise = np.random.normal(0, sigma, N)
    X = np.random.uniform(0, 3, N)
    Y = X ** 2 - 3 * X + 1 + noise  # Compute y from x
    return X, Y


def linear_regression(X, y, degree=2):
    """ Fit a linear regression equation of given degree to X, Y """
    XV = vander_matrix(X, degree)
    coeffs = np.linalg.inv(np.dot(XV.transpose(), XV)).dot(np.dot(XV.transpose(), y))
    fit_y = predict(X, coeffs)
    mse = np.sum((fit_y - y) ** 2)
    return coeffs, mse


def ridge_regression(X, Y, degree=2, lambda_p=0):
    """
    Fit a ridge regression equation of given degree to X, Y
    :param lambda_p : regularization parameter
    :param degree : degree of polynomial fit
    """
    XV = vander_matrix(X, degree)
    XVTXV = np.dot(XV.transpose(), XV)
    coeffs = np.linalg.inv(XVTXV + lambda_p * np.eye(XVTXV.shape[0])).dot(np.dot(XV.transpose(), Y))
    fit_y = predict(X, coeffs)
    mse = np.sum((fit_y - Y) ** 2)/len(Y)
    return coeffs, mse


def predict(X, coeff):
    """ Fit X to the coefficients computed after fitting polynomial"""
    X = vander_matrix(X, len(coeff) - 1)
    Y = np.dot(X, coeff)
    return Y


def cross_validate(X, y, coeff, sigma):
    """
    Algorithm :
    - Generate test data according to the given sigma
    - Compute the mse for test data
    - if test_mse > train_mse : overfit else underfit
    """
    y_fitted = predict(X, coeff)
    # The errors has been normalized to length so that lengthy vectors do not have higher errors
    train_error = np.sum((y - y_fitted) ** 2) / len(X)

    # Generating 300 sample points from the same distribution to check the fit
    test_X, test_y = generate_data(300, sigma)
    test_y_fitted = predict(test_X, coeff)
    test_error = np.sum((test_y - test_y_fitted) ** 2) / len(test_y)

    # If the differences are minimal | below a certain threshold consider it appropriate fit
    fitting_type = "Overfit"
    if test_error - train_error <= 0.1:
        fitting_type = "Appropriate"
    elif test_error < train_error:
        fitting_type = 'Underfit'

    return fitting_type, train_error, test_error


if __name__ == '__main__':
    # Parameter defining data and the equation we're fitting
    degree = 9  # Which degree of equation to fit the data
    N = [15, 100]
    sigmas = [0.05]
    # lambdas = range(0, 10)  # reguarlization parameter
    # lambdas = np.linspace(0, 100, 1000)  # Tricks used to look for underfits
    # lambdas = [0, 2, 2.5]
    # lambdas = [0, 0.5, 2.5]
    # lambdas = [0, 0.1, 3]
    # lambdas = [1, 2, 3]
    # lambdas = [4, 5, 6]
    lambdas = [0, 7.8, 3]

    # Make a list of types of data
    color = ['black', 'green', 'red', 'grey']

    # Legends
    patch = [
        mpatches.Patch(color=color[0], label='Original'),
        mpatches.Patch(color=color[3], label='9 Deg lambda {}'.format(lambdas[0])),
        mpatches.Patch(color=color[1], label='9 Deg lambda {}'.format(lambdas[1])),
        mpatches.Patch(color=color[2], label='9 Deg lambda {}'.format(lambdas[2]))]

    # Loop over all the types of dataset we need to fit ridge regression
    for num_point in N:
        for sigma in sigmas:
            X, y = generate_data(num_point, sigma)
            X, y = np.sort(np.array([X, y]))  # Sort the generated data points

            plt.scatter(X, y, color=color[0], linewidths=0.01)
            plt.legend(handles=patch)

            for index, l in enumerate(lambdas):
                data_description = "N : {} Sigma : {}".format(num_point, sigma)
                plt.title('Fitting polynomial eq to {}'.format(data_description))

                coeff, mse = ridge_regression(X, y, degree, lambda_p=l)
                plt.plot(X, predict(X, coeff), color=color[index + 1])

                fitting_type, train_error, test_error = cross_validate(X, y, coeff, sigma)
                summary = ",{}, {}, {}, {}, {}, {}, {}, {}" \
                    .format(data_description, degree, l, fitting_type, np.round(mse, 3),
                            np.round(train_error, 3), np.round(test_error, 3), np.round(coeff, 3))
                print(l, np.round(train_error - test_error, 3), summary + "\n")
                logging.info(summary)

            plt.show()
            plt.pause(1)
