"""
Generating data and adding gaussian noise to it.
And applying linear regression to it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

np.random.seed(3)

# Define the parameters of the logger - filename, loggername, mode of writing, level of logging
log_filename = 'regression.csv'
logging.basicConfig(filename=log_filename, filemode='a', level=logging.CRITICAL)
logging.info(",Data, Poly_Degree, Fit, MSE, TRAIN_ERROR_PER_PREDICTION, TEST_ERROR_PER_PREDICTION, COEFFS")


def vander_matrix(X, degree=4):
    assert degree > 0
    assert len(X) > 0

    X = np.asarray(X)
    order = degree + 1
    v = np.empty((len(X), order))
    v[:, degree] = 1
    for i in range(0, degree):
        v[:, i] = X ** (degree - i)

    # print("Self : ", np.round(v, 2))
    # print("Numpy :", np.round(np.vander(X, degree), 2))
    # print()
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

            # plt.scatter(x, y)
            # plt.title(
            #     'Data generated with Num Points : {} Noise with 0 mean and Sigma {}'.format(num_points, sigma_value))
            # plt.show()
            # plt.pause(1)

    return gen_data


def linear_regression(X, y, degree=2):
    XV = vander_matrix(X, degree)
    coeffs = np.linalg.inv(np.dot(XV.transpose(), XV)).dot(np.dot(XV.transpose(), y))
    fit_y = fit(X, coeffs)
    mse = np.sum((fit_y - y) ** 2)
    return coeffs, mse


def fit(X, weight):
    X = vander_matrix(X, len(weight) - 1)
    Y = np.dot(X, weight)
    return Y


def check_fit(train_X, train_y, test_X, test_y, weight):
    train_y_fitted = fit(train_X, weight)
    test_y_fitted = fit(test_X, weight)

    # The errors has been normalized to length so that lengthy vectors do not have higher errors
    train_error = np.sum((train_y - train_y_fitted) ** 2) / len(train_X)
    test_error = np.sum((test_y - test_y_fitted) ** 2) / len(test_y)

    model = "Overfit"
    if test_error < train_error:
        model = "Underfit"

    return model, train_error, test_error


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
    data = generate_data()
    weights = []
    degrees = [1, 2, 9]

    train_data, test_data = test_and_train_data(data)

    N = [15, 100]
    sigma = [0.01, 0.05, 0.2]
    properties = []
    for num_point in N:
        for sigma_value in sigma:
            properties.append("N : {} Sigma : {}".format(num_point, sigma_value))

    color = ['black', 'red', 'green', 'orange']

    for i, [X, y] in enumerate(data):
        plt.scatter(X, y, color='black')

        patch = [
            mpatches.Patch(color='black', label='Original'),
            mpatches.Patch(color='red', label='1 Deg'),
            mpatches.Patch(color='green', label='2 Deg'),
            mpatches.Patch(color='orange', label='9 Deg')]

        plt.legend(handles=patch)
        plt.title('Fitting polynomial eq to {}'.format(properties[i]))

        for index, degree in enumerate(degrees):
            weight, mse = linear_regression(X, y, degree)

            print(np.polyfit(X, y, degree))
            print(weight)
            print()

            weights.append(weight)
            plt.scatter(X, fit(X, weight), c=color[index + 1], label='1')
            fitting, train_error, test_error = check_fit(X, y, test_data[i][0], test_data[i][1], weight)
            print("Dataset : {}, Degree : {}, Fit : {}, MSE : {}, Coeffs : {} \tTrain Error {} Test Error : {}".format(i, degree,
            fitting, np.round(mse,3), np.round(weight,3), np.round(train_error,3),np.round(test_error,3)))
            logging.info(
            ",{}, {}, {}, {}, {}, {}, {}".format(properties[i], degree,
            fitting, np.round(mse,3), np.round(train_error,3),np.round(test_error,3), np.round(weight,3)))

            # plt.show()
            # plt.pause(1)
            print()
        break
