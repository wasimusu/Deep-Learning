from sklearn import datasets
import matplotlib.pyplot as plt
import random
import numpy as np


# Rotate it in three set directions
# Translate it horizontal and vertical direction

class OCRModel():
    def __init__(self):
        # The digits dataset
        self.digits = datasets.load_digits()
        # random.shuffle(self.digits)  # Shuffle digits inplace and returns none

    def trainTestSplit(self, ratio=0.8):
        """
        :param ratio: ratio of train data to test data
        :return:
        """
        len_train = int(len(self.digits.images) * ratio)

        self.train_images = self.digits.images[:len_train]
        self.train_labels = self.digits.target[:len_train]

        self.test_images = self.digits.images[len_train:]
        self.test_labels = self.digits.target[len_train:]

        print("Total Digits : ", len(self.digits.images))
        print("Training Data : ", len(self.train_labels))
        print("Test Data : ", len(self.test_labels))

        del self.digits
        assert len(self.train_images) == len(self.train_labels)
        assert len(self.test_images) == len(self.test_labels)

    def train(self):
        self.models = {}
        self.digit_count = {}

        digit = np.zeros((8, 8), np.float128)

        for i in range(10):
            self.models[i] = digit
            self.digit_count[i] = 0

        for index, [image, label] in enumerate(zip(self.train_images, self.train_labels)):
            digit = np.asarray(image, np.float128)
            self.models[int(label)] = self.models[label] + digit
            self.digit_count[label] += 1

        # Average the sum of accumulated pixels
        for i in range(10):
            self.models[i] = self.models[i] / self.digit_count[i]
            plt.imshow(self.models[i])
            plt.show()
            # plt.pause(1)

    def accuracy(self):
        """ Measure the accuracy of the trained model """
        correct = 0
        for index, [image, label] in enumerate(zip(self.test_images, self.test_labels)):
            distance = [0] * 10
            digit = np.asarray(image, np.float64)
            for i in range(10):
                distance[i] = np.sqrt(np.sum(self.models[i] - digit) ** 2)

            if distance.index(max(distance)) == int(label):
                correct += 1

        print("Accuracy : ", correct / len(self.test_labels) * 100)

    def inference(self):
        pass


if __name__ == '__main__':
    model = OCRModel()
    model.trainTestSplit()
    model.train()
