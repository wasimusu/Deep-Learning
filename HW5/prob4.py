import os

import torch
import torchvision
import torch.nn as nn
import cv2

# Gather all the images
# Also has random images
dir = "./data/AlexNet/"
filenames = os.listdir(dir)
filenames = [os.path.join(dir, filename) for filename in filenames]

# Initiate the model with the pretrained weights
alexnet = torchvision.models.alexnet(pretrained=True)

for filename in filenames:
    image = torch.tensor(cv2.imread(filename, cv2.IMREAD_COLOR)).view(1, 3, 227, 227).float()

    # Do not compute derivatives
    with torch.no_grad():

        # 4.2 Reading out from a layer
        layer1 = alexnet.features[0](image).squeeze(0)
        # The shape of the output from first layer is [1, 64, 56, 56]

        # Showing the first channel of the 64 channel output
        channel1 = layer1[0].numpy()
        cv2.imshow("Channel1 ", channel1)
        cv2.waitKey(1000)

        # 4.3 Output of the final layer
        output = alexnet(image)
        # The shape of output from the final layer is [1, 1000]
        class_id = torch.argmax(output, dim=1)

        print("File : {} \t\tClass : {}\tFirst Layer shape : {} ".format(os.path.split(filename)[1], class_id.item(),                                                                       layer1.size()))