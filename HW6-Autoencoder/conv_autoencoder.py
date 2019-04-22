" Undercomplete Convolutional Autoencoder "

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Number of filter in each convolution layer
        first_in = 1
        second_in = 16
        third_in = 8
        third_out = 4

        self.conv1 = nn.Conv2d(in_channels=first_in, out_channels=second_in, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=second_in, out_channels=third_in, kernel_size=(5, 5), stride=1)
        self.conv3 = nn.Conv2d(in_channels=third_in, out_channels=third_out, kernel_size=(2, 2), stride=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        code = self.conv1(x)
        code, indices1 = self.maxpool(code)

        code = self.conv2(code)
        code, indices2 = self.maxpool(code)

        code = self.conv3(code)
        code, indices3 = self.maxpool(code)

        return code, [indices1, indices2, indices3]


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Number of filter in each convolution layer
        first_in = 4
        second_in = 8
        third_in = 16
        third_out = 1

        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.tconv1 = nn.ConvTranspose2d(in_channels=first_in, out_channels=second_in, kernel_size=2, stride=1)
        self.tconv2 = nn.ConvTranspose2d(in_channels=second_in, out_channels=third_in, kernel_size=5, stride=1)
        self.tconv3 = nn.ConvTranspose2d(in_channels=third_in, out_channels=third_out, kernel_size=5, stride=1)

    def forward(self, code, indices):
        output = self.maxunpool(code, indices[2])
        output = self.tconv1(output)

        output = self.maxunpool(output, indices[1])
        output = self.tconv2(output)

        output = self.maxunpool(output, indices[0])
        output = self.tconv3(output)

        return output


class UndercompleteAutoencoder(nn.Module):
    def __init__(self):
        super(UndercompleteAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input):
        code, indices = self.encoder(input)
        recovered_input = self.decoder(code, indices)
        return recovered_input


def train(train_mode=True):
    transformer = transforms.Compose([
        torchvision.transforms.Pad((2, 2), 0),
        torchvision.transforms.ToTensor(),
    ])

    device = ("cuda" if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    filename = "model/undercomplete_autoencoder_5"
    reuse_model = True
    learning_rate = 0.01
    momentum = 0.1
    delta_loss = 0.001  # Threshold loss difference between consecutive epochs to stop training

    # Dataset for training, validation and test set
    trainset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=True, train=True)
    trainset, validationset = torch.utils.data.random_split(trainset, [50000, 10000])
    testset = torchvision.datasets.MNIST(root='./data', transform=transformer, download=True, train=False)

    # Data loader for train, test and validation set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=True)
    validationloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2, shuffle=False)

    # Defining optimizer and criterion (loss function), optimizer and model
    model = UndercompleteAutoencoder()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Use pretrained model or train new
    if reuse_model == True:
        if os.path.exists(filename):
            model.load_state_dict(torch.load(f=filename))
        else:
            print("No pre-trained model detected. Starting fresh model training.")
    model.to(device)

    if train_mode:
        # Train the model and periodically compute loss and accuracy on test set
        cur_epoch_loss = 10
        prev_epoch_loss = 20
        epoch = 1
        while abs(prev_epoch_loss - cur_epoch_loss) >= delta_loss:
            epoch_loss = 0
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(device)
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

            # Save the model every n epochs
            if epoch == 1 or epoch == 3 or epoch == 5:
                torch.save(model.state_dict(), f="{}_{}".format(filename, epoch))
                with torch.no_grad():
                    for data in validationloader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        output = model(inputs).cpu()
                        output = output.view(10, 32, 32)
                        for i, image in enumerate(output):
                            plt.imsave("images/{}_{}.jpg".format(i, epoch), image)
                        break

            epoch += 1  # Incremenet the epoch counter
            prev_epoch_loss = cur_epoch_loss
            cur_epoch_loss = epoch_loss


if __name__ == '__main__':
    train()
