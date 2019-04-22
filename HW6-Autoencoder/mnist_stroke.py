import urllib.request
import os
import tarfile
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()
device = ('cuda' if not use_cuda else 'cpu')
batch_size = 16
num_layers = 1
learning_rate = 0.05
hidden_size = 200
l2_norm = 0.0  # 0.25
momentum = 0.0
filename = "model/stroke"
delta_loss = 0.01
reuse_model = True
dropout_rate = 0.1


def getAccuracy(model, dataLoader):
    """ Compute accuracy for given dataset """
    total, correct = 0, 0
    for i in range(len(dataLoader)):
        with torch.no_grad():
            inputs, lens, targets = dataLoader.next()
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.to(device)
                targets = targets.to(device)

            # inputs = inputs.squeeze(1)

            if inputs.size(0) != batch_size: continue

            outputs = model(inputs, lens)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == targets).data.to('cpu').numpy()

            total += batch_size
            correct += score

    accuracy = correct * 1.0 / total
    assert total != 0
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

    def process(self):
        self.processed_dir = "data/processed_mnist"
        self.processed_inputs = os.path.join(self.processed_dir, "inputs.npy")
        self.processed_targets = os.path.join(self.processed_dir, "targets.npy")

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        else:
            if os.path.exists(self.processed_inputs) and os.path.exists(self.processed_targets):
                self.targets = np.load(self.processed_targets)
                self.inputs = np.load(self.processed_inputs)

                # There are only 70000
                if self.mode == "train":
                    start = 0
                    end = 1000
                elif self.mode == "test":
                    start = 50000
                    end = 51000
                else:
                    start = 60000
                    end = 61000

                self.targets = self.targets[start:end]
                self.inputs = self.inputs[start:end]

                self.num_batches = len(self.targets) // self.batch_size
                self.indices = list(range(len(self.targets)))
                self.indices = self.indices[:self.num_batches * self.batch_size]

                print("Total number of samples : ", len(self.indices))

        # If you have not already processed the sequence files, you can process them now
        self.sequence_dir = "/home/wasim/Documents/sequences"
        if not os.path.exists(self.sequence_dir):
            raise ValueError("The file of sequences does not occur at ", self.sequence_dir)

        files = os.listdir(self.sequence_dir)
        files = [file for file in files if file.__contains__("targetdata")]

        self.inputs, self.targets = [], []
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
            self.targets.append(label)
            self.inputs.append(input)

        np.save(self.processed_targets, self.targets)
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

        self.targets = self.targets[start:end]
        self.inputs = self.inputs[start:end]

        self.num_batches = len(self.targets) // self.batch_size
        self.indices = list(range(len(self.targets)))
        self.indices = self.indices[:self.num_batches * self.batch_size]

    def __next__(self):
        """ Return the next batch of data for training """
        batch_indices = [self.indices.pop(0) for _ in range(self.batch_size)]
        inputs = [self.inputs[index] for index in batch_indices]
        targets = [self.targets[index] for index in batch_indices]

        input_lens = []
        if self.batch_size > 1:
            # Sorting everything according to decrease length of sequence
            input_lens = [len(seq) for seq in inputs]

            # input_lens = sorted(input_lens, reverse=True)
            # ranks = np.argsort(input_lens)
            # inputs = [inputs[rank] for rank in ranks]
            # targets = [targets[rank] for rank in ranks]

            inputs = [input[:min(input_lens)] for input in inputs]
            inputs = self.pad(inputs)

        # # Convert the array into tensors
        inputs = torch.tensor(inputs).reshape(self.batch_size, -1, 4)
        # inputs = torch.tensor(inputs).reshape(self.batch_size, 1, -1)
        targets = torch.tensor(targets)

        if len(self.indices) < self.batch_size:
            self.indices = list(range(len(self.targets)))

        return inputs, input_lens, targets

    def next(self):
        return self.__next__()

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
        self.input_size = input_size

        self.features = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                bidirectional=bidirectional,
                                num_layers=num_layers,
                                batch_first=True)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 80),
            nn.Linear(80, num_classes)
        )

        self.hidden = self.init_hidden()

    def forward(self, x, lens):
        # Required shape of input for LSTM : (seq_len, batch, input_size)

        x = x.view(self.batch_size, -1, self.input_size)

        x, self.hidden = self.features(x, self.hidden)
        # print("X : ", x.size())
        x = x.contiguous()
        x = x[:, -1, :]  # The last hidden state
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return x

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.float).to(
                device),
            torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.float).to(
                device))


def train(train_mode=False):
    # Defining optimizer and criterion (loss function), optimizer and model
    model = MnistStrokeClassifier(hidden_size=hidden_size, batch_size=batch_size, num_layers=num_layers)
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
    # trainLoader = MnistStrokeSequence(mode="train", shuffle=True, batch_size=batch_size)
    testLoader = MnistStrokeSequence(mode="test", shuffle=True, batch_size=batch_size)

    if train_mode == True:
        # Train the model and periodically compute loss and accuracy on test set
        cur_epoch_loss = 10
        prev_epoch_loss = 20
        epoch = 1
        while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
            epoch_loss = 0
            for i in range(len(testLoader)):
                inputs, lens, targets = testLoader.next()
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                # inputs = inputs.squeeze(1)
                if inputs.size(0) != batch_size:
                    print("Breaking")
                    break

                output = model(inputs, lens)
                # print("Output : ", output)
                optimizer.zero_grad()
                loss = criterion(output, targets)

                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss

                if i % 100 == 0:
                    print(i, len(testLoader), "%.3f" % epoch_loss, "%.3f" % getAccuracy(model, testLoader))

                if i % 25 == 0:
                    print(i, len(testLoader), "%.3f" % epoch_loss)
                    epoch_loss = 0

                    prev_epoch_loss = cur_epoch_loss
                    cur_epoch_loss = epoch_loss

            print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

            # Every ten epochs compute validation accuracy
            if epoch % 2 == 0:
                print("{} Epoch. Accuracy on validation set : {}".format(epoch,
                                                                         "%.3f" % getAccuracy(model, testLoader)))

            # Save the model every ten epochs
            if epoch % 1 == 0:
                torch.save(model.state_dict(), f=filename)

            epoch += 1  # Incremenet the epoch counter

    # Do inference on test set
    print("Accuracy on test set : {}".format("%.4f" % getAccuracy(model, testLoader)))


if __name__ == '__main__':
    train(train_mode=True)

    # testLoader = MnistStrokeSequence(mode="test", shuffle=True, batch_size=64)
    # for _ in range(2):
    #     for i in range(len(testLoader)):
    #         inputs, lens, targets = testLoader.next()
    #         print(i, len(testLoader), targets)
