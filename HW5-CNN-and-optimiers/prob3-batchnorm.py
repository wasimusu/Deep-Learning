"""
Comparing the performance of Optimizers.
    - SGD
    - SGD with momentum
    - Adam
    - Adagrad
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 30 epoch
# 30 epoch


# Softmax requires especially low learning rates
learning_rate = 0.005
l2 = 0.5
batch_size = 64
dropout = 0.0
reuse_model = True
delta_loss = 0.0001  # The minimum threshold differences required between two consecutive epoch to continue training

device = ('cuda:0' if torch.cuda.is_available() else 'cpu:0')


def getAccuracy(model, dataLoader):
    """ Compute accuracy for given dataset """
    total, correct = 0, 0
    for i, data in enumerate(dataLoader):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == labels).data.to('cpu').numpy()

            total += batch_size
            correct += score

    accuracy = correct * 1.0 / total
    return accuracy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax()
        self.b1 = nn.BatchNorm1d(100)
        self.b2 = nn.BatchNorm1d(100)

    def forward(self, x):
        # Assuming the image is grayscale image
        x = x.view(-1, 28 * 28)

        x = F.relu(self.b1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.b2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x


def train(optimizer='SGD', momentum=0.0):
    filename = "model/prob3-2-model-batchnorm-{}-moment-{}-lr-{}".format(optimizer, momentum, 0.1)

    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))]
    )

    # Dataset for training, validation and test set
    trainset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=True)
    trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
    testset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=False)

    # Data loader for train, test and validation set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)
    validationloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

    # Use pretrained model or train new
    model = Net()
    if reuse_model == True:
        if os.path.exists(filename):
            model.load_state_dict(torch.load(f=filename))
            print("Loading pretrained network")
        else:
            print("No pre-trained model detected. Starting fresh model training.")

    model.to(device)

    # Defining optimizer and criterion (loss function)
    if optimizer.upper() == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer.upper() == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.upper() == 'ADAGRAD':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("We're only considering SGD, Adam and AdaGrad for now.")

    criterion = nn.CrossEntropyLoss()

    # Train the model and periodically compute loss and accuracy on test set
    cur_epoch_loss = 10
    prev_epoch_loss = 20
    epoch = 1
    while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
        epoch_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)

            model.zero_grad()
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss

        # Every ten epochs compute validation accuracy
        if epoch % 10 == 0:
            print("{}, {}, {}".format(epoch, "%.3f" % epoch_loss, "%.3f" % getAccuracy(model, validationloader)))

        else:
            print("{}, {}".format(epoch, "%.3f" % epoch_loss))

        # Save the model every ten epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f=filename)
            print()

        epoch += 1  # Increment the epoch counter
        prev_epoch_loss = cur_epoch_loss
        cur_epoch_loss = epoch_loss

        # Do not train any of the network for more than certain epoch
        if epoch == 200:
            break

    # Do inference on test set
    print("Accuracy on test set : {}".format("%.4f" % getAccuracy(model, testloader)))


if __name__ == '__main__':
    train(optimizer='sgd', momentum=0.1)
