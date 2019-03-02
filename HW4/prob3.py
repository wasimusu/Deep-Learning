"""
MNIST being trained by using Fully connected layer
Concepts used :
 - Using pretrained networks
 - Dropout
 - l1 regularization
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

learning_rate = 0.01
momentum = 0.9
num_epochs = 10
batch_size = 32
dropout = 0.2
filename = "model"
reuse_model = True

l2 = 0.99
regularization = 'l1'
if regularization == 'l1': l2 = 0

device = ('cuda:0' if torch.cuda.is_available() else 'cpu:0')


class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 100),
            F.relu(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 10),
            F.relu(),
        )

    def forward(self, x):
        # Assuming the image is grayscale image
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return x

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

# Loader for training and test set
trainset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=True, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=True, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

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

    print(epoch, "%.2f" % epoch_loss)

torch.save(model.state_dict(), f=filename)

# Check test accuracy
total, correct = 0, 0
for i, data in enumerate(testloader):
    with torch.no_grad():
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        score = sum(outputs == labels)

        total += batch_size
        correct += score

print(total, correct, total * 1.0 / correct)
