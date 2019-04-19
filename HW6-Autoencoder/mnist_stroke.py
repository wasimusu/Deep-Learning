import urllib.request
import os
import gzip
import tarfile

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = ('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.01
l2_norm = 0.1
momentum = 0
filename = "model/stroke"
delta_loss = 0.01
reuse_model = True


class MnistStrokeSequence:
    def __init__(self, train=True, shuffle=True, batch_size=1, root_dir="data"):
        """
        :param train: true if data is to be used for training
        :param shuffle: randomly shuffle the data or not
        :param batch_size:
        :param root_dir: the directory to the save the downloaded data or where the data is already saved
        """
        self.train = train
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.root_dir = root_dir

        self.url_sequence = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
        self.sequence_fname = "sequences.tar.gz"
        self.sequence_fname = os.path.join(self.root_dir, self.sequence_fname)

        # self.url_digit_thinned = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/digit-images-thinned.tar.gz"
        # self.digit_fname = "digit-images-thinned.tar.gz"
        # self.digit_fname = os.path.join(self.root_dir, self.digit_fname)

        self.maybe_download()

    def __next__(self):
        self.inputs = np.load(self.processed_inputs)
        self.labels = np.load(self.processed_labels)

        return self.inputs[0], self.labels[0]

    def process(self, save_dir=""):
        self.processed_dir = "data/processed_mnist"
        self.processed_inputs = os.path.join(self.processed_dir, "inputs.npy")
        self.processed_labels = os.path.join(self.processed_dir, "labels.npy")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        else:
            if os.path.exists(self.processed_inputs) and os.path.exists(self.processed_labels):
                print("Processed files already exist")
                return

        # If you have not already processed the sequence files, you can process them now
        self.sequence_dir = "/home/wasim/Documents/sequences"
        if not os.path.exists(self.sequence_dir):
            raise ValueError("The file of sequences does not occur at ", self.sequence_dir)

        files = os.listdir(self.sequence_dir)
        files = [file for file in files if file.__contains__("targetdata")]
        print("Total number of sample digits : ", len(files))

        inputs, labels = [], []
        for fname in files[:10]:
            fname = os.path.join(self.sequence_dir, fname)
            sequence = open(fname, mode='r', encoding='utf8').read().splitlines()
            sequence = " ".join(sequence).split()
            sequence = np.asarray(sequence, np.int8).reshape(-1, 14)
            label = sequence[:, :10]
            label = np.mean(label, axis=0)
            assert sum(label) == 1.0
            label = np.argsort(label.tolist())[-1]

            input = sequence[:, 10:]
            labels.append(label)
            inputs.append(input)

        np.save(self.processed_labels, labels)
        np.save(self.processed_inputs, inputs)

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

        # if not os.path.exists(self.digit_fname):
        #     print("Downloading ... ", self.digit_fname)
        #     urllib.request.urlretrieve(self.url_digit_thinned, self.digit_fname)


class MnistStrokeSequenceClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, batch_size=64, bidirectional=False):
        super(MnistStrokeSequenceClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.batch_size = batch_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=bidirectional,
                          num_layers=num_layers)

        self.hidden = self.init_hidden()

    def forward(self, x):
        x = x.view(1, self.batch_size, -1)
        x, self.hidden = self.lstm(x, self.hidden)

        x = x.view(-1, self.hidden_dim)
        x = self.classifier(x)
        x = F.softmax(x)
        return x

    def init_hidden(self):
        return torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)


def getAccuracy(model, dataLoader):
    """ Compute accuracy for given dataset """
    total, correct = 0, 0
    for i, data in enumerate(dataLoader):
        with torch.no_grad():
            inputs, labels = data
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


def train():
    # Defining optimizer and criterion (loss function), optimizer and model
    model = MnistStrokeSequenceClassifier()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Use pretrained model or train new
    if reuse_model == True:
        if os.path.exists(filename):
            model.load_state_dict(torch.load(f=filename))
        else:
            print("No pre-trained model detected. Starting fresh model training.")
    model.to(device)

    # Train the model and periodically compute loss and accuracy on test set
    cur_epoch_loss = 10
    prev_epoch_loss = 20
    epoch = 1
    while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data

            inputs = inputs.to(device).squeeze(1)
            labels = labels.to(device)

            if inputs.size(0) != batch_size: continue

            output = model(inputs)

            model.zero_grad()
            loss = criterion(output, labels)

            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss

        print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

        # Every ten epochs compute validation accuracy
        if epoch % 10 == 0:
            print("{} Epoch. Accuracy on validation set : {}".format(epoch,
                                                                     "%.3f" % getAccuracy(model, validationloader)))

        # Save the model every ten epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f=filename)
            print()

        epoch += 1  # Incremenet the epoch counter
        prev_epoch_loss = cur_epoch_loss
        cur_epoch_loss = epoch_loss

    # Do inference on test set
    print("Accuracy on test set : {}".format("%.4f" % getAccuracy(model, testloader)))


if __name__ == '__main__':
    train_data = MnistStrokeSequence(train=True, shuffle=True)
    train_data.process()
    inputs, labels = train_data.next()
    print(inputs, labels)
