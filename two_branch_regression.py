import torch
import torch.nn as nn


# ... (other imports) ...

class FeatureExtractionBranch(nn.Module):
    def __init__(self, in_channels=1, initial_filters=32):
        super(FeatureExtractionBranch, self).__init__()
        # Ensure 'inplace=True' is removed from ReLU if you plan to debug with hooks,
        # otherwise it's generally fine.
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, initial_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(),  # Removed inplace=True for robustness, can add back if desired
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(initial_filters * 2, initial_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(initial_filters * 4, initial_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(initial_filters * 8, initial_filters * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(initial_filters * 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.ReLU(),  # Removed inplace=True
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),  # Removed inplace=True
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid() # Add this if your target alpha is strictly [0,1]
        )

    def forward(self, x):
        return self.fc_layers(x)


class TwoBranchRegressionModel(nn.Module):
    def __init__(self, initial_filters_per_branch=32, input_image_size=(256, 256)):  # Add input_image_size here
        super(TwoBranchRegressionModel, self).__init__()

        self.bleed_branch = FeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)
        self.source_branch = FeatureExtractionBranch(in_channels=1, initial_filters=initial_filters_per_branch)

        # --- MODIFICATION STARTS HERE ---
        # Use a dummy input with batch_size > 1 to avoid BatchNorm error during initialization
        dummy_batch_size = 2  # Or any number > 1
        dummy_input_bleed = torch.zeros(dummy_batch_size, 1, *input_image_size)  # Use *input_image_size

        # Temporarily set branches to eval mode to avoid running BatchNorm in training mode with dummy data
        # This is a robust way to handle this, as BatchNorm won't try to compute stats.
        self.bleed_branch.eval()
        with torch.no_grad():  # Ensure no gradients are computed for this dummy pass
            dummy_output_bleed = self.bleed_branch(dummy_input_bleed)
        self.bleed_branch.train()  # Set back to train mode

        # Assuming both branches output features of the same shape
        branch_output_spatial_dims = dummy_output_bleed.shape[2:]
        branch_output_channels = dummy_output_bleed.shape[1]

        regression_head_input_size = (branch_output_channels * 2) * branch_output_spatial_dims[0] * \
                                     branch_output_spatial_dims[1]

        self.regression_head = RegressionHead(input_feature_size=regression_head_input_size)
        # --- MODIFICATION ENDS HERE ---

    def forward(self, x):
        bleed_input = x[:, 0:1, :, :]
        source_input = x[:, 1:2, :, :]

        features_bleed = self.bleed_branch(bleed_input)
        features_source = self.source_branch(source_input)

        fused_features = torch.cat((features_bleed, features_source), dim=1)

        output = self.regression_head(fused_features)
        return output
