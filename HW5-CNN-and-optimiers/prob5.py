"""
MNIST being trained by using CNN
Concepts used :
 - Using pretrained networks
 - l2 regularization
 - Hyper-parameter tuning
 - Saving and restoring model
 - Moving models to gpu or cpu
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Softmax requires especially low learning rates
learning_rate = 0.05
momentum = 0.0
num_epochs = 20
batch_size = 64
dropout = 0.0
reuse_model = True
delta_loss = 0.0001  # The minimum threshold differences required between two consecutive epoch to continue training

l2 = 0.99
l1 = 0.99
regularization = 'l2'
if regularization == 'l1': l2 = 0

device = ('cuda:0' if torch.cuda.is_available() else 'cpu:0')

filename = "model/model-softmax-64-20"    # Best
filename = "model/model-softmax-20-64"


def getAccuracy(dataLoader):
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
        out_channel1 = 20
        out_channel2 = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, out_channel1, (5, 5), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channel1, out_channel2, (3, 3), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.classifier = nn.Linear(6 * 6 * out_channel2, 10)

    def forward(self, x):
        x = self.features(x)

        # Images are not being processed alone, they're processed in batches
        # x.size(0) is batch_size

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.softmax(x)
        return x


transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# Dataset for training, validation and test set
trainset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=True)
trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
testset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=False)

num_train_samples = trainset.__len__()
l1 = l1 * 1.0 / num_train_samples

# Data loader for train, test and validation set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)
validationloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

# Use pretrained model or train new
model = Net()
if reuse_model == True:
    if os.path.exists(filename):
        model.load_state_dict(torch.load(f=filename))
    else:
        print("No pre-trained model detected. Starting fresh model training.")

model.to(device)

# Defining optimizer and criterion (loss function)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
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

        # Only used for L1 regularization
        if regularization == "l1":
            for param in model.parameters():
                loss += l1 * torch.sum(torch.abs(param)).data.to('cpu').numpy()

        loss.backward()
        optimizer.step()

        epoch_loss += loss

    print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

    # Every ten epochs compute validation accuracy
    if epoch % 10 == 0:
        print("{} Epoch. Accuracy on validation set : {}".format(epoch, "%.3f" % getAccuracy(validationloader)))

    # Save the model every ten epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f=filename)
        print()

    epoch += 1  # Incremenet the epoch counter
    prev_epoch_loss = cur_epoch_loss
    cur_epoch_loss = epoch_loss

# Do inference on test set
print("Accuracy on test set : {}".format("%.4f" % getAccuracy(testloader)))
