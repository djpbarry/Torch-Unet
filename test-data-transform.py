import os
import random
import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# Define a custom dataset class for handling crosstalk data
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, label_dir):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.label_dir = label_dir
        self.mixed_channel_filenames = sorted([f for f in os.listdir(mixed_channel_dir) if f.endswith('.tif')])
        self.pure_source_filenames = sorted([f for f in os.listdir(pure_source_dir) if f.endswith('.tif')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        if not (len(self.mixed_channel_filenames) == len(self.pure_source_filenames) == len(self.label_filenames)):
            raise ValueError("Number of mixed channel, pure source, and label images must be the same.")

    def __len__(self):
        return len(self.mixed_channel_filenames)

    def __getitem__(self, idx):
        mixed_channel_path = os.path.join(self.mixed_channel_dir, self.mixed_channel_filenames[idx])
        pure_source_path = os.path.join(self.pure_source_dir, self.pure_source_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)
        label_image_np = iio.imread(label_path).astype(np.float32)

        return mixed_image_np, source_image_np, label_image_np

def get_train_transforms(target_size):
    def train_transforms_fn(mixed_np, source_np, label_np):
        # Convert numpy arrays to tensors
        mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
        source_tensor = torch.from_numpy(source_np).unsqueeze(0)
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        # Random horizontal flip
        if torch.rand(1) < 0.5:
            mixed_tensor = TF.hflip(mixed_tensor)
            source_tensor = TF.hflip(source_tensor)
            label_tensor = TF.hflip(label_tensor)

        # Random vertical flip
        if torch.rand(1) < 0.5:
            mixed_tensor = TF.vflip(mixed_tensor)
            source_tensor = TF.vflip(source_tensor)
            label_tensor = TF.vflip(label_tensor)

        # # Random rotation
        # angle = T.RandomRotation.get_params([-10, 10])
        # mixed_tensor = TF.rotate(mixed_tensor, angle)
        # source_tensor = TF.rotate(source_tensor, angle)
        # label_tensor = TF.rotate(label_tensor, angle)

        # Resize tensors
        resize_transform = T.Resize(target_size)
        mixed_tensor = resize_transform(mixed_tensor)
        source_tensor = resize_transform(source_tensor)
        label_tensor = resize_transform(label_tensor)

        return mixed_tensor, source_tensor, label_tensor

    return train_transforms_fn

def save_transformed_images(dataset, transform_fn, output_dir, num_samples=5):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Randomly select a subset of indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        mixed_image, source_image, label_image = dataset[idx]

        # Apply transformations
        mixed_tensor, source_tensor, label_tensor = transform_fn(mixed_image, source_image, label_image)

        # Save transformed images as TIFF files
        iio.imwrite(os.path.join(output_dir, f"mixed_transformed_{idx}.tiff"),mixed_tensor.squeeze().numpy())
        iio.imwrite(os.path.join(output_dir, f"source_transformed_{idx}.tiff"),source_tensor.squeeze().numpy())
        iio.imwrite(os.path.join(output_dir, f"label_transformed_{idx}.tiff"),label_tensor.squeeze().numpy())

if __name__ == "__main__":
    # Define data directories
    mixed_channel_data_dir = "H:/GitRepos/Python/IDRTrainingData/crosstalk_training_data/bleed"
    pure_source_data_dir = "H:/GitRepos/Python/IDRTrainingData/crosstalk_training_data/source"
    label_data_dir = "H:/GitRepos/Python/IDRTrainingData/crosstalk_training_data/ground_truth"
    target_size = (256, 256)

    # Create dataset
    dataset = CrosstalkDataset(mixed_channel_data_dir, pure_source_data_dir, label_data_dir)

    # Get the transformation function
    transform_fn = get_train_transforms(target_size)

    # Apply transformations and save the results
    save_transformed_images(dataset, transform_fn, "transformed_images", num_samples=5)
