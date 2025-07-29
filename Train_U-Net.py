import csv
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from unet import UNet


# Define a custom dataset class for handling crosstalk data
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, label_dir, transform=None):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mixed_channel_filenames = sorted([f for f in os.listdir(mixed_channel_dir) if f.endswith('.tif')])
        self.pure_source_filenames = sorted([f for f in os.listdir(pure_source_dir) if f.endswith('.tif')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        if not (len(self.mixed_channel_filenames) == len(self.pure_source_filenames) == len(self.label_filenames)):
            raise ValueError("Number of mixed channel, pure source, and label images must be the same.")

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
        return len(self.mixed_channel_filenames)

    def __getitem__(self, idx):
        mixed_channel_path = os.path.join(self.mixed_channel_dir, self.mixed_channel_filenames[idx])
        pure_source_path = os.path.join(self.pure_source_dir, self.pure_source_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)
        label_image_np = iio.imread(label_path).astype(np.float32)

        if self.transform:
            mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, label_image_np)
        else:
            raise ValueError("No transform pipeline provided for dataset.")

        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


def get_train_transforms(target_size):
    def train_transforms_fn(mixed_np, source_np, label_np):
        mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
        source_tensor = torch.from_numpy(source_np).unsqueeze(0)
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        if torch.rand(1) < 0.5:
            mixed_tensor = TF.hflip(mixed_tensor)
            source_tensor = TF.hflip(source_tensor)
            label_tensor = TF.hflip(label_tensor)

        if torch.rand(1) < 0.5:
            mixed_tensor = TF.vflip(mixed_tensor)
            source_tensor = TF.vflip(source_tensor)
            label_tensor = TF.vflip(label_tensor)

        # angle = T.RandomRotation.get_params([-10, 10])
        # mixed_tensor = TF.rotate(mixed_tensor, angle)
        # source_tensor = TF.rotate(source_tensor, angle)
        # label_tensor = TF.rotate(label_tensor, angle)

        # resize_transform = T.Resize(target_size)
        # mixed_tensor = resize_transform(mixed_tensor)
        # source_tensor = resize_transform(source_tensor)
        # label_tensor = resize_transform(label_tensor)

        return mixed_tensor, source_tensor, label_tensor

    return train_transforms_fn


def get_val_test_transforms(target_size):
    def val_test_transforms_fn(mixed_np, source_np, label_np):
        mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
        source_tensor = torch.from_numpy(source_np).unsqueeze(0)
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        # resize_transform = T.Resize(target_size)
        # mixed_tensor = resize_transform(mixed_tensor)
        # source_tensor = resize_transform(source_tensor)
        # label_tensor = resize_transform(label_tensor)

        return mixed_tensor, source_tensor, label_tensor

    return val_test_transforms_fn


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    log_file_path = "training_log.csv"
    file_exists = os.path.isfile(log_file_path)

    with open(log_file_path, mode='a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        if not file_exists:
            log_writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.shape != labels.shape:
                    outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                        align_corners=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_train_loss = running_loss / len(train_dataloader.dataset)
            print(f"Epoch {epoch + 1} Train Loss: {epoch_train_loss:.6f}")

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(inputs)
                    if outputs.shape != labels.shape:
                        outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                            align_corners=False)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

            epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
            print(f"Epoch {epoch + 1} Validation Loss: {epoch_val_loss:.6f}")

            log_writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss])
            log_file.flush()

    print("Training complete. Losses logged to training_log.csv.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mixed_channel_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/bleed"
    pure_source_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/source"
    label_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/ground_truth"

    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    U_NET_IN_CHANNELS = 2
    U_NET_OUT_CHANNELS = 1
    TARGET_IMAGE_SIZE = (256, 256)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    if not (abs(TRAIN_RATIO + VAL_RATIO) < 1.0):
        print("Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0.")

    model = UNet(in_channels=2, out_channels=1, init_features=32)

    print("\nCreating dataset instances for initial file listing...")
    try:
        temp_dataset = CrosstalkDataset(
            mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
            transform=get_val_test_transforms(TARGET_IMAGE_SIZE)
        )
        print(f"Total samples found in directories: {len(temp_dataset)}")
    except Exception as e:
        print(f"Error initializing temporary dataset: {e}")
        exit()

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

            mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, label_image_np)
            input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
            return input_tensor, label_tensor


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

    train_dataloader = DataLoader(
        train_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True
    )

    print("Dataloaders created for training, validation, and testing.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training with validation...")
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS, device)
    print("Training finished!")

    model_save_path = "crosstalk_detection_unet_2input_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = UNet(in_channels=2, out_channels=1, init_features=32)
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
