from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        # Initialize the number of features
        features = init_features

        # Encoder path with four levels of downsampling
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling using max pooling
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer at the deepest part of the network
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # Decoder path with four levels of upsampling
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )  # Upsampling using transposed convolution
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        # Final convolution layer to produce the output
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # Encoder path forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck forward pass
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path with skip connections from the encoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection from enc4 to dec4
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection from enc3 to dec3
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection from enc2 to dec2
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection from enc1 to dec1
        dec1 = self.decoder1(dec1)

        # Final convolution layer with sigmoid activation for binary segmentation
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        # Define a basic block consisting of two convolutional layers with batch normalization and ReLU activation
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),  # First convolutional layer
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),  # Batch normalization
                    (name + "relu1", nn.ReLU(inplace=True)),  # ReLU activation
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),  # Second convolutional layer
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),  # Batch normalization
                    (name + "relu2", nn.ReLU(inplace=True)),  # ReLU activation
                ]
            )
        )

# Example usage of the UNet class
# model = UNet(in_channels=3, out_channels=1, init_features=32)
# input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
# output = model(input_tensor)
# print(output.shape)
