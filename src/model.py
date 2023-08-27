import torch.nn as nn
import torch.nn.functional as F
from constant import *

## Here, you should build the neural network. A simple model is given below as an example, 
##you can modify the neural network architecture.

class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Define encoder layers
        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        # Convolutional Layer 1 (Encoder): Processes input data, extracts and highlights features.
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Convolutional Layer 2 (Encoder): Further processes features, generates new feature maps.

        # Define decoder layers
        self.decoder_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # Convolutional Layer 1 (Decoder): Processes encoder features, aims to reconstruct original features.
        self.decoder_conv2 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        # Convolutional Layer 2 (Decoder): Produces final output, including class predictions.

        # Upsample layer
        self.upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        # Upsample Layer: Resizes the output to the desired input size.

    def forward(self, x):
        # Encoder Convolutional Layer 1
        x = self.encoder_conv1(x)
        x = F.relu(x)
        # ReLU Activation after Encoder Conv1

        # Encoder Convolutional Layer 2
        x = self.encoder_conv2(x)
        x = F.relu(x)
        # ReLU Activation after Encoder Conv2

        # Decoder Convolutional Layer 1
        x = self.decoder_conv1(x)
        x = F.relu(x)
        # ReLU Activation after Decoder Conv1

        # Decoder Convolutional Layer 2
        x = self.decoder_conv2(x)
        x = F.softmax(x, dim=1)
        # Softmax Function for Decoder Output

        # Upsample
        x = self.upsample(x)
        
        return x

if __name__ == '__main__':
    model = FoInternNet(input_size=(HEIGHT, WIDTH), n_classes=N_CLASS)
