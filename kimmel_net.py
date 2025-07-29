import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 224, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(224, 112, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened features
        conv_output_size = self._get_conv_output((256, 256))

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _get_conv_output(self, shape):
        # Helper method to calculate the size of the flattened features
        with torch.no_grad():
            input = torch.zeros(1, 2, *shape)  # Use 2 channels for input
            output = self.conv_layers(input)
            return int(torch.prod(torch.tensor(output.size()[1:])))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Example usage
# model = RegressionModel()
# input_tensor = torch.randn(1, 2, 256, 256)  # Example input tensor with 2 channels and size 256x256
# output = model(input_tensor)
# print(output)
