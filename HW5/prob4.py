import os

import torch
import torchvision
import cv2

# Gather all the images
# Also has random images
dir = "./data/AlexNet/"
filenames = os.listdir(dir)
filenames = [os.path.join(dir, filename) for filename in filenames]

# Initiate the model with the pretrained weights
alexnet = torchvision.models.alexnet(pretrained=True)

# Apply these transformations for all the images in the dataset
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

testset = torchvision.datasets.ImageFolder(dir, transform=transform)
data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

for i, data in enumerate(data_loader):
    inputs, labels = data

    # Do not compute derivatives
    with torch.no_grad():
        # 4.2 Reading out from a layer
        layer1 = alexnet.features[0](inputs).squeeze(0)
        # The shape of the output from first layer is [1, 64, 56, 56]

        # Showing the first channel of the 64 channel output
        channel1 = layer1[2].numpy() * 255
        cv2.imshow("Channel1 ", channel1)
        cv2.waitKey(1000)

        # Save some channel of the image as output
        cv2.imwrite(os.path.join("data/results/", i.__str__() + ".jpg"), channel1)

        # 4.3 Output of the final layer
        output = alexnet(inputs)
        # The shape of output from the final layer is [1, 1000]
        class_id = torch.argmax(output, dim=1)

        print("File : {} \t\tClass : {}\tFirst Layer shape : {} ".format(i, class_id.item(), layer1.size()))
