""" Compare a few machine learning methods on given dataset for HW2 """
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read data file and parse it
file = open('Data/data_seed.dat', mode='r').readlines()
data = [line.split() for line in file]
data = [float(num) for row in data for num in row]
data = np.asarray(data).reshape(-1, 8)

np.random.seed(1)  # For repeatable experiments

class TestClassifiers():
    def __init__(self, method='svm', n_neighbors=5, class_weight='balanced'):
        self.method = method
        # Dictionary of all the methods that we're going to try and compare
        self.classifier_map = {'svm': svm.SVC(gamma='scale'),
                               'logistic_regression': LogisticRegression(class_weight=class_weight),
                               'knn': KNeighborsClassifier(n_neighbors=n_neighbors)
                               }

    def cross_validation(self, k=5):
        """
        Apply five fold cross validation
        Leave one out cross validation has k = N

        Algorithm:
            Divide the data into k parts
            where
                training_data = (k-1) part
                validation_data = 1 part
                train on the training_data and report error on validation_data

        k = -1 for leave one out cross validation
        """
        test_errors, train_errors = [], []

        # For leave out cross validation k = n
        if k == -1:
            k = len(data)

        for _ in range(k):
            shuffled_data = data.copy()

            # Do not shuffle data for leave one out cross validation
            if k != -1:
                # Copy the original data and shuffle it
                np.random.shuffle(shuffled_data)

            # Divide it into k folds
            split = int(((k - 1) / k) * len(data))
            train_data = shuffled_data[:split]
            test_data = shuffled_data[split:]

            # Find train_X, train_Y, test_X, test_Y
            train_X = np.asarray(train_data[:, :6])
            train_Y = np.asarray(train_data[:, 7])

            test_X = np.asarray(test_data[:, :6])
            test_Y = np.asarray(test_data[:, 7])

            # Use (k-1) part for training and 1 part for testing
            self.fit(train_X, train_Y)
            test_error = self.compute_error(test_X, test_Y)
            train_error = self.compute_error(train_X, train_Y)
            test_errors.append(test_error)
            train_errors.append(train_error)

        # Average the error
        avg_train_error = np.round(np.average(np.asarray(train_errors), axis=0), 3)
        avg_test_error = np.round(np.average(np.asarray(test_errors), axis=0), 3)
        print("The average error of {} is - Train : {}\tTest : {} Overfit : {}".format(self.method,
                                                                                       avg_train_error, avg_test_error,
                                                                                       avg_test_error > avg_train_error))
        return avg_test_error, avg_train_error

    def fit(self, X, Y):
        """ Fit the training data to the classifier model and compute accuracy """
        self.classifier = self.classifier_map[self.method]
        self.classifier.fit(X, Y)

    def compute_error(self, X, Y):
        """ Compute accuracy given the test data """
        accuracy = self.classifier.score(X, Y)
        error = 1 - accuracy
        # print("Accuracy of {} : {}".format(self.method, np.round(accuracy)))
        return error


if __name__ == '__main__':
    color = ['green', 'red', 'grey']

    # Legends
    patch = [
        mpatches.Patch(color=color[0], label='Train Error'),
        mpatches.Patch(color=color[1], label='Test Error')]

    # # Prob 2.2
    # test_errors = []
    # train_errors = []
    # ks = [1, 5, 10, 15]
    #
    # # Five fold cross validation : KNN
    # for k in ks:
    #     print(k)
    #     knn = TestClassifiers(method='knn', n_neighbors=k)
    #     test_error, train_error = knn.cross_validation()
    #     test_errors.append(test_error)
    #     train_errors.append(train_error)
    #
    # plt.legend(handles=patch)
    # plt.plot(ks, test_errors, color=color[0])
    # plt.plot(ks, train_errors, color=color[1])
    #
    # plt.title('test error as a function of k with 5 fold cross validation')
    # plt.xlabel('k')
    # plt.ylabel('Test Error')
    # plt.show()
    #
    # # Leave one out cross validation
    # test_errors = []
    # train_errors = []
    # for k in ks:
    #     print(k)
    #     knn = TestClassifiers(method='knn', n_neighbors=k)
    #     test_error, train_error = knn.cross_validation(k=-1)
    #     test_errors.append(test_error)
    #     train_errors.append(train_error)
    #
    # plt.legend(handles=patch)
    # plt.plot(ks, test_errors, color=color[0])
    # plt.plot(ks, train_errors, color=color[1])
    #
    # plt.title('test error as a function of k with leave one out cross validation')
    # plt.xlabel('k')
    # plt.ylabel('Test Error')
    # plt.show()

    # Prob 2.3
    # Tuning svm
    methods = ['svm', 'logistic_regression']
    class_weights = [None, 'balanced']
    for method in methods:
        test_errors = []
        train_errors = []
        for class_weight in class_weights:
            test_classifiers = TestClassifiers(method=method, class_weight=class_weight)
            test_error, train_error = test_classifiers.cross_validation()
            test_errors.append(test_error)
            train_errors.append(train_error)

        plt.legend(handles=patch)
        plt.plot([0, 1], test_errors, color=color[0])
        plt.plot([0, 1], train_errors, color=color[1])

        plt.title('Error as a function of balanced and unbalanced weights of classes in {}'.format(method))
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.show()
