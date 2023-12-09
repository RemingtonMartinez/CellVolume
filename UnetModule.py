import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Change the number of input channels to 1
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu4 = nn.ReLU()
        
        # Add more convolutional layers as needed for your U-Net architecture
        # ...

        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.relu5 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.relu6 = nn.ReLU()
        self.conv6 = nn.Conv2d(16, 1, 3, padding=1)  # Change the number of output channels to 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        x3 = self.maxpool(x2)
        x4 = self.conv3(x3)
        x4 = self.relu3(x4)
        x5 = self.conv4(x4)
        x5 = self.relu4(x5)
        
        # Add more layers and skip connections for the U-Net architecture
        # ...

        x6 = self.upconv1(x5)
        x6 = self.relu5(x6)
        x7 = self.conv5(torch.cat([x6, x2], dim=1))  # Concatenate skip connection from x2
        x7 = self.relu6(x7)
        x8 = self.conv6(x7)
        output = self.sigmoid(x8)

        return output
