import os

import torch
import torchvision
import matplotlib.pyplot as plt

# Apply these transformations for all the images in the dataset
# Convert image to tensor
# Normalize the image with given mean and standard deviation
# Center crop the image with the given dimension
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     # torchvision.transforms.Normalize((0.5,), (0.5,)),
     # torchvision.transforms.CenterCrop((227, 227)),
     ]
)

# Read all the images
dir = "./data/AlexNet/"
testset = torchvision.datasets.ImageFolder(dir, transform=transform)
data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# Initiate the model with the pretrained weights
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()  # Set the model to evaluation model so that dropout is not applied

print("File ,\tClass, \t\t\t\t\tFirst Layer shape ")
for i, data in enumerate(data_loader):
    inputs, labels = data

    # Do not compute derivatives
    with torch.no_grad():
        # 4.2 Reading out from a layer
        layer1 = alexnet.features[0](inputs).squeeze(0)
        # The shape of the output from first layer is [1, 64, 56, 56]

        # Showing the first channel of the 64 channel output
        channel1 = layer1[2].numpy()
        plt.imshow(channel1)
        plt.show()

        # Save some channel of the image as output
        filename = os.path.join("data/results/", i.__str__() + ".jpg")
        plt.imsave(filename, channel1)

        # 4.3 Output of the final layer
        output = alexnet(inputs)
        # The shape of output from the final layer is [1, 1000]
        score, class_id = torch.topk(output, dim=1, k=5)

        print("{}, \t\t{},\t{}".format(i, class_id.numpy().ravel(), layer1.size()))
