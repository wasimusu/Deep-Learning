""" MNIST being trained by using Fully connected layer"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.01
momentum = 0.9
num_epochs = 10
batch_size = 32

device = ('cuda:0' if torch.cuda.is_available() else 'cpu:0')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

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
trainset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', transform=transforms, download=False, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

# Network, Optimizer and loss function
net = Net().to(device)
# How to apply l2 regularization to the weights and biases
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# TODO : What's pytorch way of applying L2 norm regularization
# TODO : How do you custom initialize weights in nn.Linear()
# TODO : Save and restore model

# Train the model and periodically compute loss and accuracy on test set
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = net(inputs)

        net.zero_grad()
        loss = criterion(output, labels)
        epoch_loss += loss
        loss.backward()
        optimizer.step()

    print(epoch, "%.2f" % epoch_loss)

    # Check test accuracy
    for i, data in enumerate(testloader):
        total, correct = 0, 0
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            outputs = torch.argmax(outputs, dim=1)
            score = sum(outputs == labels)

            total += batch_size
            correct += score
            print(score, batch_size)
            break
