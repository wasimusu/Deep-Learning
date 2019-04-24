import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os

# Parameters for the Simple Autoencoder and training
device = ("cuda" if torch.cuda.is_available() else 'cpu')
batch_size = 10
filename = "model/simple_ae"
reuse_model = True
learning_rate = 0.01
momentum = 0.1


class SimpleAE(nn.Module):
    def __init__(self, image_size=28 * 28, hidden_nodes=4, batch_size=16):
        super(SimpleAE, self).__init__()
        self.encoder = nn.Linear(in_features=image_size, out_features=hidden_nodes)
        self.decoder = nn.Linear(in_features=hidden_nodes, out_features=image_size)
        self.batch_size = batch_size

    def forward(self, input):
        input = input.view(self.batch_size, 1, -1)
        code = self.encoder(input)
        recovered_input = self.decoder(code)
        recovered_input = recovered_input.view(self.batch_size, 28, -1)
        return recovered_input


def train(train_mode=True, hidden_nodes=4):
    transformer = transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    # Dataset for training and test set
    trainset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=False, train=True)
    trainset, _ = torch.utils.data.random_split(trainset, [50000, 10000])
    testset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=False, train=False)

    # Data loader for train, test and validation set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2, shuffle=False)

    # Defining optimizer and criterion (loss function), optimizer and model
    model = SimpleAE(batch_size=batch_size, hidden_nodes=hidden_nodes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Use pretrained model or train new
    if reuse_model == True:
        if os.path.exists("{}_{}".format(filename, hidden_nodes)):
            model.load_state_dict(torch.load(f="{}_{}".format(filename, hidden_nodes)))
        else:
            print("No pre-trained model detected. Starting fresh model training.")
    model.to(device)

    if train_mode:
        # Train the model and periodically compute loss and accuracy on test set
        for epoch in range(30):
            epoch_loss = 0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                output = model(inputs)

                model.zero_grad()
                inputs = inputs.squeeze(1).view(batch_size, -1)
                output = output.view(batch_size, -1).float()
                assert inputs.shape == output.shape
                loss = criterion(output, inputs)

                loss.backward()
                optimizer.step()

                epoch_loss += loss

            print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

            # Save the model every ten epochs
            torch.save(model.state_dict(), f="{}_{}".format(filename, hidden_nodes))

    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            output = model(inputs).cpu()

            # Save the autoencoded images
            output = output.view(10, 28, 28)
            for i, image in enumerate(output):
                plt.imsave("SimpleAE/{}_{}.jpg".format(labels[i], hidden_nodes), image)

            # Save the original images
            original = inputs.view(10, 28, 28).cpu()
            for i, image in enumerate(original):
                plt.imsave("SimpleAE/{}_original.jpg".format(labels[i]), image)
            break


if __name__ == '__main__':
    train(train_mode=True, hidden_nodes=4)
    train(train_mode=True, hidden_nodes=8)
    train(train_mode=True, hidden_nodes=16)
