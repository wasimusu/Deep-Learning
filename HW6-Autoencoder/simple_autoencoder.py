import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

import os


class Encoder(nn.Module):
    def __init__(self, input_size=29 * 28, output_size=4, batch_size=2):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder = nn.Linear(input_size, output_size)

    def forward(self, input):
        input = input.view(batch_size, 1, -1)
        code = self.encoder(input)
        return code


class Decoder(nn.Module):
    def __init__(self, input_size=4, output_size=28 * 28, batch_size=2):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, input):
        recovered_original = self.decoder(input)
        recovered_original = recovered_original.view(self.batch_size, 28, -1)
        return recovered_original


class Autoencoder(nn.Module):
    def __init__(self, image_size=28 * 28, hidden_nodes=4):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size=image_size, output_size=hidden_nodes)
        self.decoder = Decoder(input_size=hidden_nodes, output_size=image_size)

    def forward(self, input):
        code = self.encoder(input)
        recovered_input = self.decoder(code)
        return recovered_input


transformer = transforms.Compose([
    torchvision.transforms.Pad((0, 0), 0),
    torchvision.transforms.ToTensor(),
])

device = ("cuda" if torch.cuda.is_available() else 'cpu')
batch_size = 2
filename = ""
reuse_model = True
learning_rate = 0.01
momentum = 0.1
delta_loss = 0.001  # Threshold loss difference between consecutive epochs to stop training

# Dataset for training, validation and test set
trainset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=False, train=True)
trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
testset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=False, train=False)

# Data loader for train, test and validation set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)
validationloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)

# Defining optimizer and criterion (loss function), optimizer and model
model = Autoencoder()
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
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)

        model.zero_grad()
        inputs = inputs.squeeze(1).view(batch_size, -1)
        output = output.view(batch_size, -1).float()
        assert inputs.shape == output.shape
        # print("output shape : {},\tinput shape : {}".format(output.shape, inputs.shape))
        loss = criterion(output, inputs)

        loss.backward()
        optimizer.step()

        epoch_loss += loss

    print("{} Epoch. Loss : {}".format(epoch, "%.3f" % epoch_loss))

    # Save the model every ten epochs
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f=filename)
        print()

    epoch += 1  # Incremenet the epoch counter
    prev_epoch_loss = cur_epoch_loss
    cur_epoch_loss = epoch_loss
