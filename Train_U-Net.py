# Import necessary libraries
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Define a custom dataset class for handling crosstalk data
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, label_dir, transform=None):
        # Initialize directories and transformation
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.label_dir = label_dir
        self.transform = transform

        # List all .tif files in the directories
        self.mixed_channel_filenames = sorted([f for f in os.listdir(mixed_channel_dir) if f.endswith('.tif')])
        self.pure_source_filenames = sorted([f for f in os.listdir(pure_source_dir) if f.endswith('.tif')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        # Check if the number of files in each directory matches
        if not (len(self.mixed_channel_filenames) == len(self.pure_source_filenames) == len(self.label_filenames)):
            raise ValueError("Number of mixed channel, pure source, and label images must be the same.")

        # Check if filenames match across directories
        for i in range(len(self.mixed_channel_filenames)):
            mixed_base = '_'.join(self.mixed_channel_filenames[i].split('_')[:-1])
            source_base = '_'.join(self.pure_source_filenames[i].split('_')[:-1])
            label_base = '_'.join(self.label_filenames[i].split('_')[:-2])
            if not (mixed_base == source_base == label_base):
                raise ValueError(
                    f"Filename mismatch at index {i}: "
                    f"Mixed: {self.mixed_channel_filenames[i]}, "
                    f"Source: {self.pure_source_filenames[i]}, "
                    f"Label: {self.label_filenames[i]}"
                )
        print(f"Found {len(self.mixed_channel_filenames)} matching samples.")

    def __len__(self):
        # Return the total number of samples
        return len(self.mixed_channel_filenames)

    def __getitem__(self, idx):
        # Load images from the directories
        mixed_channel_path = os.path.join(self.mixed_channel_dir, self.mixed_channel_filenames[idx])
        pure_source_path = os.path.join(self.pure_source_dir, self.pure_source_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        # Read images and convert to numpy arrays
        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)
        label_image_np = iio.imread(label_path).astype(np.float32)

        # Convert numpy arrays to PIL images
        mixed_image_pil = T.ToPILImage()(mixed_image_np)
        source_image_pil = T.ToPILImage()(source_image_np)
        label_image_pil = T.ToPILImage()(label_image_np)

        # Apply transformations if specified
        if self.transform:
            mixed_tensor, source_tensor, label_tensor = self.transform(
                mixed_image_pil, source_image_pil, label_image_pil
            )
        else:
            raise ValueError("No transform pipeline provided for dataset.")

        # Concatenate mixed and source tensors along the channel dimension
        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


# Function to train the model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Loop over the dataset multiple times (based on the number of epochs)
    for epoch in range(num_epochs):
        # Set the model to training mode (enables dropout, batch normalization, etc.)
        model.train()

        # Initialize running loss to accumulate the loss over the batches
        running_loss = 0.0

        # Iterate over data batches in the training dataloader
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            # Move input data and labels to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients to prevent accumulation from previous batches
            optimizer.zero_grad()

            # Perform the forward pass: compute predictions by passing inputs through the model
            outputs = model(inputs)

            # Resize model outputs to match label dimensions if they don't match
            if outputs.shape != labels.shape:
                outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                    align_corners=False)

            # Calculate the loss between the predicted outputs and the true labels
            loss = criterion(outputs, labels)

            # Perform the backward pass: compute gradients of the loss with respect to model parameters
            loss.backward()

            # Update the model parameters using the computed gradients
            optimizer.step()

            # Accumulate the loss for the current batch, scaled by the batch size
            running_loss += loss.item() * inputs.size(0)

        # Calculate the average training loss for the epoch
        epoch_train_loss = running_loss / len(train_dataloader.dataset)

        # Print the average training loss for the current epoch
        print(f"Epoch {epoch + 1} Train Loss: {epoch_train_loss:.6f}")

        # Set the model to evaluation mode (disables dropout, uses population statistics for batch normalization, etc.)
        model.eval()

        # Initialize running validation loss
        val_running_loss = 0.0

        # Disable gradient computation for validation to save memory and computations
        with torch.no_grad():
            # Iterate over data batches in the validation dataloader
            for inputs, labels in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                # Move input data and labels to the specified device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Perform the forward pass: compute predictions by passing inputs through the model
                outputs = model(inputs)

                # Resize model outputs to match label dimensions if they don't match
                if outputs.shape != labels.shape:
                    outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                        align_corners=False)

                # Calculate the loss between the predicted outputs and the true labels
                loss = criterion(outputs, labels)

                # Accumulate the validation loss for the current batch, scaled by the batch size
                val_running_loss += loss.item() * inputs.size(0)

        # Calculate the average validation loss for the epoch
        epoch_val_loss = val_running_loss / len(val_dataloader.dataset)

        # Print the average validation loss for the current epoch
        print(f"Epoch {epoch + 1} Validation Loss: {epoch_val_loss:.6f}")


# Function to get training transformations
def get_train_transforms(target_size):
    def train_transforms_fn(mixed_pil, source_pil, label_pil, current_target_size):
        # Random horizontal flip
        if torch.rand(1) < 0.5:
            mixed_pil = TF.hflip(mixed_pil)
            source_pil = TF.hflip(source_pil)
            label_pil = TF.hflip(label_pil)
        # Random vertical flip
        if torch.rand(1) < 0.5:
            mixed_pil = TF.vflip(mixed_pil)
            source_pil = TF.vflip(source_pil)
            label_pil = TF.vflip(label_pil)
        # Random rotation
        angle = T.RandomRotation.get_params([-10, 10])
        mixed_pil = TF.rotate(mixed_pil, angle)
        source_pil = TF.rotate(source_pil, angle)
        label_pil = TF.rotate(label_pil, angle)
        # Color jitter
        # color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        # mixed_pil = color_jitter(mixed_pil)
        # source_pil = color_jitter(source_pil)
        # Final transform to resize and convert to tensor
        final_transform = T.Compose([
            T.Resize(current_target_size),
            T.ToTensor()
        ])
        mixed_tensor = final_transform(mixed_pil)
        source_tensor = final_transform(source_pil)
        label_tensor = final_transform(label_pil)
        return mixed_tensor, source_tensor, label_tensor

    return lambda mixed_pil, source_pil, label_pil: train_transforms_fn(mixed_pil, source_pil, label_pil, target_size)


# Function to get validation and test transformations
def get_val_test_transforms(target_size):
    final_transform_default = T.Compose([
        T.Resize(target_size),
        T.ToTensor()
    ])
    return lambda mixed_pil, source_pil, label_pil: (
        final_transform_default(mixed_pil),
        final_transform_default(source_pil),
        final_transform_default(label_pil)
    )


if __name__ == "__main__":
    # Set device to GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data directories
    mixed_channel_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/bleed"
    pure_source_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/source"
    label_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/ground_truth"

    # Define hyperparameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    U_NET_IN_CHANNELS = 2
    U_NET_OUT_CHANNELS = 1
    TARGET_IMAGE_SIZE = (256, 256)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Check if ratios sum to 1
    if not (abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6):
        print(
            "Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0. Data might be left out or sizes might be adjusted.")

    # Load U-Net model from PyTorch Hub
    print("\nLoading U-Net model from PyTorch Hub...")
    try:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=U_NET_IN_CHANNELS, out_channels=U_NET_OUT_CHANNELS,
                               init_features=32, pretrained=False)
        print("U-Net model loaded successfully with 2 input channels.")
    except Exception as e:
        print(f"Error loading U-Net from PyTorch Hub: {e}")
        print(
            "Please ensure you have internet connectivity or that 'mateuszbuda/brain-segmentation-pytorch' is cached locally by PyTorch Hub.")
        exit()

    # Create a temporary dataset to get all filenames
    print("\nCreating dataset instances for initial file listing...")
    try:
        temp_dataset = CrosstalkDataset(
            mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
            transform=get_val_test_transforms(TARGET_IMAGE_SIZE)  # A dummy transform, just for init
        )
        print(f"Total samples found in directories: {len(temp_dataset)}")
    except Exception as e:
        print(f"Error initializing temporary dataset: {e}")
        exit()

    # Split data into training, validation, and test sets
    print("\nSplitting data using filename lists for correct augmentation application...")
    all_mixed_filenames = temp_dataset.mixed_channel_filenames
    all_source_filenames = temp_dataset.pure_source_filenames
    all_label_filenames = temp_dataset.label_filenames
    total_samples = len(all_mixed_filenames)
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(total_samples).tolist()
    train_size = int(TRAIN_RATIO * total_samples)
    val_size = int(VAL_RATIO * total_samples)
    test_size = total_samples - train_size - val_size
    train_indices = shuffled_indices[0:train_size]
    val_indices = shuffled_indices[train_size: train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]
    print(f"Split sizes: Train = {len(train_indices)}, Validation = {len(val_indices)}, Test = {len(test_indices)}")


    # Define a custom dataset class for handling split data
    class SplitCrosstalkDataset(Dataset):
        def __init__(self, mixed_channel_dir, pure_source_dir, label_dir, transform, indices,
                     all_mixed_filenames, all_source_filenames, all_label_filenames):
            self.mixed_channel_dir = mixed_channel_dir
            self.pure_source_dir = pure_source_dir
            self.label_dir = label_dir
            self.transform = transform
            self.indices = indices
            self.mixed_channel_filenames_split = [all_mixed_filenames[i] for i in indices]
            self.pure_source_filenames_split = [all_source_filenames[i] for i in indices]
            self.label_filenames_split = [all_label_filenames[i] for i in indices]
            for i in range(len(self.mixed_channel_filenames_split)):
                mixed_base = '_'.join(self.mixed_channel_filenames_split[i].split('_')[:-1])
                source_base = '_'.join(self.pure_source_filenames_split[i].split('_')[:-1])
                label_base = '_'.join(self.label_filenames_split[i].split('_')[:-2])
                if not (mixed_base == source_base == label_base):
                    raise ValueError(f"Internal split filename mismatch: {mixed_base}, {source_base}, {label_base}")
            print(f"SplitCrosstalkDataset created with {len(self.indices)} samples.")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx_in_split):
            mixed_channel_path = os.path.join(self.mixed_channel_dir, self.mixed_channel_filenames_split[idx_in_split])
            pure_source_path = os.path.join(self.pure_source_dir, self.pure_source_filenames_split[idx_in_split])
            label_path = os.path.join(self.label_dir, self.label_filenames_split[idx_in_split])
            mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
            source_image_np = iio.imread(pure_source_path).astype(np.float32)
            label_image_np = iio.imread(label_path).astype(np.float32)
            mixed_image_pil = T.ToPILImage()(mixed_image_np)
            source_image_pil = T.ToPILImage()(source_image_np)
            label_image_pil = T.ToPILImage()(label_image_np)
            mixed_tensor, source_tensor, label_tensor = self.transform(
                mixed_image_pil, source_image_pil, label_image_pil
            )
            input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
            return input_tensor, label_tensor


    # Create final datasets with appropriate transformations
    train_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
        transform=get_train_transforms(TARGET_IMAGE_SIZE),
        indices=train_indices,
        all_mixed_filenames=all_mixed_filenames,
        all_source_filenames=all_source_filenames,
        all_label_filenames=all_label_filenames
    )

    val_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
        transform=get_val_test_transforms(TARGET_IMAGE_SIZE),
        indices=val_indices,
        all_mixed_filenames=all_mixed_filenames,
        all_source_filenames=all_source_filenames,
        all_label_filenames=all_label_filenames
    )

    test_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
        transform=get_val_test_transforms(TARGET_IMAGE_SIZE),
        indices=test_indices,
        all_mixed_filenames=all_mixed_filenames,
        all_source_filenames=all_source_filenames,
        all_label_filenames=all_label_filenames
    )

    # Create data loaders for training, validation, and testing
    train_dataloader = DataLoader(train_dataset_final, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=os.cpu_count() // 2 or 1)
    val_dataloader = DataLoader(val_dataset_final, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=os.cpu_count() // 2 or 1)
    test_dataloader = DataLoader(test_dataset_final, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=os.cpu_count() // 2 or 1)
    print("Dataloaders created for training, validation, and testing.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("\nStarting training with validation...")
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS, device)
    print("Training finished!")

    # Save the trained model
    model_save_path = "crosstalk_detection_unet_2input_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    # Evaluate the model on the test set
    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                  in_channels=U_NET_IN_CHANNELS, out_channels=U_NET_OUT_CHANNELS,
                                  init_features=32, pretrained=False)
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    loaded_model.eval()
    loaded_model.to(device)

    test_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Test Set Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = loaded_model(inputs)
            if outputs.shape != labels.shape:
                outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                    align_corners=False)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)

    final_test_loss = test_running_loss / len(test_dataloader.dataset)
    print(f"\nFinal Test Loss: {final_test_loss:.6f}")
    print("Test set evaluation complete.")
