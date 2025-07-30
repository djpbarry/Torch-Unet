import torch
import torch.nn as nn


class FeatureExtractionBranch(nn.Module):
    def __init__(self, in_channels=1, initial_filters=32):
        super(FeatureExtractionBranch, self).__init__()
        # Example for one branch. It takes 1 channel as input.
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, initial_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 128x128

            # Block 2
            nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 64x64

            # Block 3
            nn.Conv2d(initial_filters * 2, initial_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 32x32

            # Block 4
            nn.Conv2d(initial_filters * 4, initial_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 16x16

            # Block 5
            nn.Conv2d(initial_filters * 8, initial_filters * 16, kernel_size=3, stride=1, padding=1),  # Max 512 filters
            nn.BatchNorm2d(initial_filters * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: 8x8 (for 256x256 input)
        )

    def forward(self, x):
        return self.conv_blocks(x)


class RegressionHead(nn.Module):
    def __init__(self, input_feature_size):
        super(RegressionHead, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 1)  # Final scalar output
            # For 0-1 range, consider nn.Sigmoid() here.
            # E.g., nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        )

    def forward(self, x):
        return self.fc_layers(x)


class TwoBranchRegressionModel(nn.Module):
    def __init__(self, initial_filters_per_branch=32):
        super(TwoBranchRegressionModel, self).__init__()

        # Branch for the 'bleed' channel (Input channel 0)
        self.bleed_branch = FeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)
        # Branch for the 'source' channel (Input channel 1)
        self.source_branch = FeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)

        # Calculate the size of features coming from each branch
        # Use a dummy input to determine size after feature extraction
        dummy_input_bleed = torch.zeros(1, 1, 256, 256)
        dummy_output_bleed = self.bleed_branch(dummy_input_bleed)
        # Assuming both branches output features of the same shape
        branch_output_spatial_dims = dummy_output_bleed.shape[2:]  # (8, 8) for 256x256 input
        branch_output_channels = dummy_output_bleed.shape[1]  # 512 filters

        # The input feature size for the regression head will be:
        # (channels from bleed_branch + channels from source_branch) * spatial_height * spatial_width
        regression_head_input_size = (branch_output_channels * 2) * branch_output_spatial_dims[0] * \
                                     branch_output_spatial_dims[1]

        self.regression_head = RegressionHead(input_feature_size=regression_head_input_size)

    def forward(self, x):
        # x is (batch_size, 2, 256, 256)
        # Split into two single-channel inputs
        bleed_input = x[:, 0:1, :, :]  # Take the first channel (index 0)
        source_input = x[:, 1:2, :, :]  # Take the second channel (index 1)

        # Pass through independent branches
        features_bleed = self.bleed_branch(bleed_input)
        features_source = self.source_branch(source_input)

        # Concatenate features along the channel dimension
        fused_features = torch.cat((features_bleed, features_source), dim=1)  # dim=1 for channels

        # Pass through the regression head
        output = self.regression_head(fused_features)
        return output

    # Helper function to ensure input_feature_size calculation for RegressionHead is correct
    # This is implicitly handled by the dummy input calculation in __init__
    # but if you need a separate helper similar to _get_conv_output
    # you could do something like this (though it's better integrated):
    # def _get_regression_head_input_size(self, input_shape=(256, 256)):
    #     dummy_bleed = torch.zeros(1, 1, *input_shape)
    #     dummy_source = torch.zeros(1, 1, *input_shape)
    #     out_bleed = self.bleed_branch(dummy_bleed)
    #     out_source = self.source_branch(dummy_source)
    #     fused = torch.cat((out_bleed, out_source), dim=1)
    #     return int(torch.prod(torch.tensor(fused.size()[1:])))
