"""
MNIST being trained by using Fully connected layer
Concepts used :
 - Using pretrained networks
 - Dropout
 - l1 regularization
 - l2 regularization
 - Hyper-parameter tuning
 - Saving and restoring model97
 - Moving models to gpu or cpu
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

learning_rate = 0.01
momentum = 0.9
num_epochs = 20
batch_size = 32
dropout = 0.2
filename = "model"
reuse_model = True

l2 = 0.99
regularization = 'l2'
if regularization == 'l1': l2 = 0

device = ('cuda:0' if torch.cuda.is_available() else 'cpu:0')


def getAccuracy(dataLoader):
    # Check accuracy
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
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Assuming the image is grayscale image
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


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

# Use pretrained model97 or train new
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

# Train the model97 and periodically compute loss and accuracy on test set
for epoch in range(num_epochs):
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
                loss += torch.sum(torch.abs(param))

        epoch_loss += loss
        loss.backward()
        optimizer.step()

    print("{} / {} epoch. Loss : {}".format(epoch, num_epochs, "%.2f" % epoch_loss))

    # Every two epochs compute validation accuracy
    if (epoch + 1) % 2 == 0:
        print("{} / {} Epoch. Accuracy on validation set : {}".format(epoch, num_epochs,
                                                                      "%.2f" % getAccuracy(validationloader)))

# Save the model97
torch.save(model.state_dict(), f=filename)

# Do inference on test set
print("Accuracy on test set : {}".format("%.2f" % getAccuracy(testloader)))
