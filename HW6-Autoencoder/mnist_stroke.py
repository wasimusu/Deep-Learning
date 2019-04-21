import urllib.request
import os
import tarfile
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = ('cuda' if not torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.01
l2_norm = 0.1
momentum = 0
filename = "model/stroke"
delta_loss = 0.01
reuse_model = True


def getAccuracy(model, dataLoader):
    """ Compute accuracy for given dataset """
    total, correct = 0, 0
    for i in range(len(dataLoader)):
        with torch.no_grad():
            inputs, labels = dataLoader.next()
            inputs = inputs.to(device)
            labels = labels.to(device)

            if inputs.size(0) != batch_size: continue

            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == labels).data.to('cpu').numpy()

            total += batch_size
            correct += score

    accuracy = correct * 1.0 / total
    return accuracy


class MnistStrokeSequence:
    def __init__(self, mode="test", shuffle=True, batch_size=1, root_dir="data"):
        """
        :param train: true if data is to be used for training
        :param shuffle: randomly shuffle the data or not
        :param batch_size:
        :param root_dir: the directory to the save the downloaded data or where the data is already saved
        """
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.root_dir = root_dir

        self.url_sequence = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
        self.sequence_fname = "sequences.tar.gz"
        self.sequence_fname = os.path.join(self.root_dir, self.sequence_fname)

        # self.url_digit_thinned = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/digit-images-thinned.tar.gz"
        # self.digit_fname = "digit-images-thinned.tar.gz"
        # self.digit_fname = os.path.join(self.root_dir, self.digit_fname)

        self.indices = []
        self.maybe_download()
        self.process()

    def __next__(self):
        batch_indices = [self.indices.pop(0) for _ in range(self.batch_size)]
        inputs = [self.inputs[index] for index in batch_indices]
        labels = [self.labels[index] for index in batch_indices]
        inputs = self.pad(inputs)

        inputs = torch.tensor(inputs).reshape(self.batch_size, -1, 4)
        labels = torch.tensor(labels)

        if len(self.indices) == 0:
            self.indices = list(range(len(self.labels)))

        return inputs, labels

    def next(self):
        return self.__next__()

    def process(self):
        self.processed_dir = "data/processed_mnist"
        self.processed_inputs = os.path.join(self.processed_dir, "inputs.npy")
        self.processed_labels = os.path.join(self.processed_dir, "labels.npy")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        else:
            if os.path.exists(self.processed_inputs) and os.path.exists(self.processed_labels):
                self.labels = np.load(self.processed_labels)
                self.inputs = np.load(self.processed_inputs)

                # There are only 70000
                if self.mode == "train":
                    start = 0
                    end = 50000
                elif self.mode == "test":
                    start = 50000
                    end = 60000
                else:
                    start = 60000
                    end = 70000

                self.labels = self.labels[start:end]
                self.inputs = self.inputs[start:end]

                self.num_batches = len(self.labels) // self.batch_size
                self.indices = list(range(len(self.labels)))
                self.indices = self.indices[:self.num_batches * self.batch_size]

                print("Total number of samples : ", len(self.indices))

        # If you have not already processed the sequence files, you can process them now
        self.sequence_dir = "/home/wasim/Documents/sequences"
        if not os.path.exists(self.sequence_dir):
            raise ValueError("The file of sequences does not occur at ", self.sequence_dir)

        files = os.listdir(self.sequence_dir)
        files = [file for file in files if file.__contains__("targetdata")]

        self.inputs, self.labels = [], []
        for fname in files:
            fname = os.path.join(self.sequence_dir, fname)

            sequence = open(fname, mode='r', encoding='utf8').read().splitlines()
            sequence = " ".join(sequence).split()
            sequence = np.asarray(sequence, np.int32).reshape(-1, 14)

            label = sequence[:, :10]
            label = np.mean(label, axis=0)
            assert sum(label) == 1.0
            label = np.argsort(label.tolist())[-1]

            input = sequence[:, 10:].flatten()
            self.labels.append(label)
            self.inputs.append(input)

        np.save(self.processed_labels, self.labels)
        np.save(self.processed_inputs, self.inputs)

        # There are only 70000
        if self.mode == "train":
            start = 0
            end = 50000
        elif self.mode == "test":
            start = 50000
            end = 60000
        else:
            start = 60000
            end = 70000

        self.labels = self.labels[start:end]
        self.inputs = self.inputs[start:end]

        self.num_batches = len(self.labels) // self.batch_size
        self.indices = list(range(len(self.labels)))
        self.indices = self.indices[:self.num_batches * self.batch_size]

    def maybe_download(self):
        """ Download the files if they do not exist """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if not os.path.exists(self.sequence_fname):
            print("Downloading ... ", self.sequence_fname)
            urllib.request.urlretrieve(self.url_sequence, self.sequence_fname)
            # print("Extracting tar.gz and saving them at the same location")
            # tarfile.open(self.sequence_fname).extractall(self.root_dir)

    def pad(self, l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def __len__(self):
        return self.num_batches


class MnistStrokeClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, batch_size=64, num_classes=10, bidirectional=False):
        super(MnistStrokeClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size

        self.features = nn.GRU(input_size=input_size,
                               hidden_size=hidden_size,
                               bidirectional=bidirectional,
                               num_layers=num_layers,
                               batch_first=True,
                               )

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.hidden = self.init_hidden()

    def forward(self, x):
        # Required shape of input for LSTM : (seq_len, batch, input_size)
        x, self.hidden = self.features(x, self.hidden)

        x = x.contiguous()
        x = x.mean(1)
        x = x.view(-1, self.hidden_size)

        x = self.classifier(x)

        x = F.softmax(x, dim=0)

        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size,
                           dtype=torch.float).to(device)


def train(train_mode=False):
    # Defining optimizer and criterion (loss function), optimizer and model
    model = MnistStrokeClassifier(hidden_size=100)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()  # Input : (N, C) Target : (N)

    # Use pretrained model or train new
    if reuse_model == True:
        if os.path.exists(filename):
            model.load_state_dict(torch.load(f=filename))
        else:
            print("No pre-trained model detected. Starting fresh model training.")
    model.to(device)

    # validationLoader = MnistStrokeSequence(mode="validate", shuffle=True, batch_size=batch_size)
    # trainLoader = MnistStrokeSequence(mode="tran", shuffle=True, batch_size=batch_size)
    testLoader = MnistStrokeSequence(mode="train", shuffle=True, batch_size=batch_size)

    if train_mode == True:
        # Train the model and periodically compute loss and accuracy on test set
        cur_epoch_loss = 10
        prev_epoch_loss = 20
        epoch = 1
        while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
            epoch_loss = 0
            for i in range(len(testLoader)):
                inputs, labels = testLoader.next()
                inputs = torch.tensor(inputs).to(device).float().squeeze(1)
                labels = torch.tensor(labels).to(device)

                if inputs.size(0) != batch_size: continue

                output = model(inputs)

                model.zero_grad()
                loss = criterion(output, labels)

                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss

            print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

            # Every ten epochs compute validation accuracy
            if epoch % 2 == 0:
                print("{} Epoch. Accuracy on validation set : {}".format(epoch,
                                                                         "%.3f" % getAccuracy(model, testLoader)))

            # Save the model every ten epochs
            if epoch % 1 == 0:
                torch.save(model.state_dict(), f=filename)

            epoch += 1  # Incremenet the epoch counter
            prev_epoch_loss = cur_epoch_loss
            cur_epoch_loss = epoch_loss

    # Do inference on test set
    print("Accuracy on test set : {}".format("%.4f" % getAccuracy(model, testLoader)))


if __name__ == '__main__':
    train(train_mode=True)

    # testLoader = MnistStrokeSequence(mode="test", shuffle=True, batch_size=1000)
    # for _ in range(2):
    #     for i in range(len(testLoader)):
    #         testLoader.next()
