import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dataset import MnistStrokeSequence

use_cuda = torch.cuda.is_available()
device = ('cuda' if not use_cuda else 'cpu')
batch_size = 2
num_layers = 1
learning_rate = 0.05
hidden_size = 300
l2_norm = 0.3
momentum = 0.00
filename = "model/stroke"
reuse_model = False
dropout_rate = 0.05
num_epochs = 50


# 2 - 1 - 0.05 - 200  - 0.3 - 0.00

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

            if inputs.size(0) != batch_size: continue

            outputs = model(inputs, lens)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == targets).data.to('cpu').numpy()

            total += batch_size
            correct += score

    accuracy = correct * 1.0 / total
    assert total != 0
    return accuracy


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
            nn.Linear(hidden_size * self.num_directions, num_classes),
        )

    def forward(self, x, lens):
        """
        x : input
        lens : length of each input
        """
        # Required shape of input for LSTM : (batch, seq_len, input_size)
        self.hidden = self.init_hidden()

        x = x.view(self.batch_size, -1, self.input_size)

        x, self.hidden = self.features(x, self.hidden)
        # print("X : ", x.size())
        # x = x.contiguous()
        x = x[:, -1, :]  # The last hidden state
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return x

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.float)
        c0 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, self.hidden_size, dtype=torch.float)
        return (h0.to(device), c0.to(device))


def train(train_mode=False):
    """
    :param train_mode: training mode or inference mode
    :return: None
    """
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
    trainLoader = MnistStrokeSequence(mode="train", shuffle=True, batch_size=batch_size)
    testLoader = MnistStrokeSequence(mode="test", shuffle=True, batch_size=batch_size, match_dimension="mean")

    if train_mode == True:
        # Train the model and periodically compute loss and accuracy on test set
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(len(trainLoader)):
                inputs, lens, targets = trainLoader.next()
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                if inputs.size(0) != batch_size:
                    print("Breaking due to size mismatch")
                    break

                output = model(inputs, lens)
                # print("Output : ", output)
                optimizer.zero_grad()
                loss = criterion(output, targets)

                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss

            print("{} Epoch. Loss : {}\t{}".format(epoch, "%.3f" % epoch_loss, "%.3f" % getAccuracy(model, testLoader)))

            # Save the model every ten epochs
            if epoch % 1 == 0:
                torch.save(model.state_dict(), f=filename)

    # Do inference on test set
    print("Accuracy on test set : {}".format("%.4f" % getAccuracy(model, testLoader)))


if __name__ == '__main__':
    train(train_mode=True)
