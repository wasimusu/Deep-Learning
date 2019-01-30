""" Compare a few machine learning methods on given dataset for HW2 """
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Read data file and parse it
file = open('Data/data_seed.dat', mode='r').readlines()
data = [line.split() for line in file]
data = [float(num) for row in data for num in row]
data = np.asarray(data).reshape(-1, 8)


# Write a program that applies a k-nn classiﬁer to the data with k ∈{1,5,10,15}.
# Calculate the test error using both leave-one-out validation and 5-fold cross validation.
# Plot the test error as a function of k. You may use the existing methods in scikit-learn or
# other libraries for ﬁnding the k-nearest neighbors, but do not use any built-in k-nn classiﬁers.
# Also, do not use any existing libraries or methods for cross validation.
# Do any values of k result in underﬁtting or overﬁtting?

class TestClassifiers():
    def __init__(self, method='svm', n_neighbors=5):
        self.method = method
        # Dictionary of all the methods that we're going to try and compare
        self.classifier_map = {'svm': svm.SVC(gamma='scale'),
                               'logistic_regression': LogisticRegression(solver='lbfgs', multi_class='auto',
                                                                         max_iter=200),
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
        """
        test_errors, train_errors = [], []
        for _ in range(k):
            # Copy the original data and shuffle it
            shuffled_data = data.copy()
            np.random.shuffle(shuffled_data)

            # Divide it into k folds
            split = int(((k - 1) / k) * len(data))
            train_data = data[:split]
            test_data = data[split:]

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
        print("The average error of {} is : Train : {}\tTest : {} Overfit : {}".format(self.method,
                                                                                       avg_train_error, avg_test_error,
                                                                                       avg_test_error > avg_train_error))
        return avg_test_error

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
    methods = ['logistic_regression', 'svm', 'knn']
    for method in methods:
        t = TestClassifiers(method=method)
        t.cross_validation()
