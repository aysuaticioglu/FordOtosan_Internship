import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a function named double_conv with 2 parameters: in_channels and out_channels
def double_conv(in_channels, out_channels, mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),  # 3x3 convolution layer
        nn.BatchNorm2d(mid_channels),  # Batch normalization operation
        nn.ReLU(inplace=True),  # Rectified Linear Unit (ReLU) activation function
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),  # 3x3 convolution layer
        nn.BatchNorm2d(out_channels),  # Batch normalization operation
        nn.ReLU(inplace=True)  # Rectified Linear Unit (ReLU) activation function
    )

# Define a neural network class named FoInternNet that inherits from nn.Module
class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Define the down-sampling path with double_conv layers and max-pooling
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)  # Max-pooling layer

        # Define the up-sampling path with up-sampling and double_conv layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)  # 1x1 convolution for the final output

    # Define the forward pass of the network
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)
        x = nn.Softmax(dim=1)(x)  # Apply softmax to the output

        return x
