import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

import os


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Number of filter in each convolution layer
        first_in = 1
        second_in = 16
        third_in = 8
        third_out = 4

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=first_in, out_channels=second_in, kernel_size=(5, 5), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=second_in, out_channels=third_in, kernel_size=(5, 5), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=third_in, out_channels=third_out, kernel_size=(2, 2), stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, input):
        code = self.encoder(input)
        return code


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Number of filter in each convolution layer
        first_in = 4
        second_in = 8
        third_in = 16
        third_out = 1

        self.decoder = nn.Sequential(
            # 2*2*4
            nn.MaxUnpool2d(kernel_size=2, stride=2),  # 5*5*8
            nn.ConvTranspose2d(in_channels=first_in, out_channels=second_in, kernel_size=2, stride=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),  # 5*5*8
            nn.ConvTranspose2d(in_channels=second_in, out_channels=third_in, kernel_size=2, stride=1),

            nn.MaxUnpool2d(kernel_size=2, stride=2),  # 5*5*8
            nn.ConvTranspose2d(in_channels=third_in, out_channels=third_out, kernel_size=2, stride=1),
        )

    def forward(self, code):
        estimated_original = self.decoder(code)
        return estimated_original


class UndercompleteAutoencoder(nn.Module):
    def __init__(self):
        super(UndercompleteAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        code = self.encoder(input)
        recovered_input = self.decoder(code)
        return recovered_input


transformer = transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Pad((2, 2), 0)]
)

device = ("cuda" if torch.cuda.is_available() else 'cpu')
batch_size = 1
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

# Use pretrained model or train new
model = UndercompleteAutoencoder()
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
