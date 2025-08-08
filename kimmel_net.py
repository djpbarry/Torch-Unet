import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Added Batch Normalization
            nn.LeakyReLU(0.01),   # Changed to LeakyReLU
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # Added Batch Normalization
            nn.LeakyReLU(0.01),   # Changed to LeakyReLU
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # Added Batch Normalization
            nn.LeakyReLU(0.01),   # Changed to LeakyReLU
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Added Batch Normalization
            nn.LeakyReLU(0.01),   # Changed to LeakyReLU
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling to reduce spatial dimensions to (1,1)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Dynamically calculate flattened feature size after conv and pooling
        conv_output_size = self._get_conv_output((256, 256))

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _get_conv_output(self, shape):
        # Helper to determine output feature size after conv and pooling
        with torch.no_grad():
            input = torch.zeros(1, 2, *shape)  # batch=1, channels=2, HxW
            output = self.global_pool(self.conv_layers(input))
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # flatten all but batch dimension
        x = self.fc_layers(x)
        return x
