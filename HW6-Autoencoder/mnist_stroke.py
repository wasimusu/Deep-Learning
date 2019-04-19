import urllib.request
import os
import gzip
import tarfile


class MnistStroke:
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
        self.url_sequence = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/blob/master/sequences.tar.gz"
        self.url_digit_thinned = "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/blob/master/digit-images-thinned.tar.gz"
        self.root_dir = root_dir
        self.sequence_fname = "sequences.tar.gz"
        self.digit_fname = "digit-images-thinned.tar.gz"

        self.maybe_download()

    def __next__(self):
        file = tarfile.open(self.digit_fname, mode='r').read()
        print(file)

    def next(self):
        return self.__next__()

    def maybe_download(self):
        """ Download the files if they do not exist """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.sequence_fname = os.path.join(self.root_dir, self.sequence_fname)
        self.digit_fname = os.path.join(self.root_dir, self.digit_fname)

        if not os.path.exists(self.sequence_fname):
            print("Downloading ... ", self.sequence_fname)
            urllib.request.urlretrieve(self.url_sequence, self.sequence_fname)

        if not os.path.exists(self.digit_fname):
            print("Downloading ... ", self.digit_fname)
            urllib.request.urlretrieve(self.url_digit_thinned, self.digit_fname)


import torch
import torch.nn as nn
import torch.optim as optim

device = ('cuda' if torch.cuda.is_available() else 'cpu')


class MnistStroker(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, batch_size=64, bidirectional=False):
        super(MnistStroker, self).__init__()
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
        x = x.view(-1, 1)
        encoding, self.hidden = self.gru(x, self.hidden)
        return encoding

    def init_hidden(self):
        return torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size)


def accuracy(model, data):
    pass


def train():
    train_data = MnistStroke(train=True, shuffle=True)

    model = MnistStroker()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, weight_decay=0.1)

    for epoch in range(10):
        for data in train_data:
            model.zero_grad()

            inputs, targets = data
            predictions = model(inputs)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train_data = MnistStroke(train=True, shuffle=True)
    train_data.next()
