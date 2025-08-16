import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedFeatureExtractionBranch(nn.Module):
    def __init__(self, in_channels=1, initial_filters=64): # Reduced initial_filters
        super(SimplifiedFeatureExtractionBranch, self).__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, initial_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 128x128

            # Block 2
            nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 64x64

            # Block 3 - Removed one block compared to previous version
            nn.Conv2d(initial_filters * 2, initial_filters * 4, kernel_size=3, stride=1, padding=1), # Max 64 filters
            nn.BatchNorm2d(initial_filters * 4),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 32x32

            # Potentially add one more if 3 blocks are too shallow for your features
            nn.Conv2d(initial_filters * 4, initial_filters * 8, kernel_size=3, stride=1, padding=1), # Max 128 filters
            nn.BatchNorm2d(initial_filters * 8),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output size: 16x16
        )

    def forward(self, x):
        return self.conv_blocks(x)

class SimplifiedRegressionHead(nn.Module):
    def __init__(self, input_feature_size):
        super(SimplifiedRegressionHead, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_feature_size, 512),  # Reduced from 512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4),  # Slightly reduced dropout, you can experiment

            nn.Linear(512, 128), # Reduced from 512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.4), # Slightly reduced dropout, you can experiment

            nn.Linear(128, 1), # Only one hidden FC layer
            nn.Sigmoid() # Keep Sigmoid if alpha is strictly [0,1]
        )

    def forward(self, x):
        return self.fc_layers(x)

class SimplifiedTwoBranchRegressionModel(nn.Module):
    def __init__(self, initial_filters_per_branch=16, input_image_size=(256, 256)):
        super(SimplifiedTwoBranchRegressionModel, self).__init__()

        # Use the simplified feature extraction branch
        self.bleed_branch = SimplifiedFeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)
        self.source_branch = SimplifiedFeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)

        # Calculate the size of features coming from each branch
        dummy_batch_size = 2 # Or any number > 1
        dummy_input_bleed = torch.zeros(dummy_batch_size, 1, *input_image_size)

        # Temporarily set branches to eval mode for size calculation
        self.bleed_branch.eval()
        with torch.no_grad():
            dummy_output_bleed = self.bleed_branch(dummy_input_bleed)
        self.bleed_branch.train() # Set back to train mode

        branch_output_spatial_dims = dummy_output_bleed.shape[2:] # e.g., (32, 32) if 3 blocks
        branch_output_channels = dummy_output_bleed.shape[1] # e.g., 64 filters

        regression_head_input_size = (branch_output_channels * 2) * branch_output_spatial_dims[0] * branch_output_spatial_dims[1]

        # Use the simplified regression head
        self.regression_head = SimplifiedRegressionHead(input_feature_size=regression_head_input_size)

    def forward(self, x):
        # x is (batch_size, 2, 256, 256)
        # Split into two single-channel inputs
        bleed_input = x[:, 0:1, :, :]
        source_input = x[:, 1:2, :, :]

        # Pass through independent branches
        features_bleed = self.bleed_branch(bleed_input)
        features_source = self.source_branch(source_input)

        # Concatenate features along the channel dimension
        fused_features = torch.cat((features_bleed, features_source), dim=1)

        # Pass through the regression head
        output = self.regression_head(fused_features)
        return output * 0.5
