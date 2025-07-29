import csv
import os
import re  # Import regex for pattern matching

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# --- Custom Dataset Classes (MODIFIED for scalar labels from filename and improved matching) ---
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, transform=None):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.transform = transform

        # Regex to extract the unique ID (e.g., '3126239') and the alpha value (e.g., '0.49')
        # This pattern works for both '_mixed.tif' and '_source.tif' files
        self.file_pattern = re.compile(r'image_(\d+)_alpha_(\d+\.?\d*)_(mixed|source)\.tif')

        # Store mappings: {image_id: {alpha: value, mixed_file: path, source_file: path}}
        all_sample_info = {}

        # Process mixed channel files
        for filename in os.listdir(mixed_channel_dir):
            if filename.endswith('.tif'):
                match = self.file_pattern.search(filename)
                if match:
                    image_id = match.group(1)
                    alpha_value = float(match.group(2))
                    file_type = match.group(3)  # 'mixed' or 'source'

                    if file_type == 'mixed':
                        if image_id not in all_sample_info:
                            all_sample_info[image_id] = {'alpha': alpha_value, 'mixed_file': filename}
                        else:
                            # Basic consistency check: alpha should be same for the same ID
                            if all_sample_info[image_id]['alpha'] != alpha_value:
                                print(
                                    f"Warning: Alpha mismatch for ID {image_id} (mixed). Existing: {all_sample_info[image_id]['alpha']}, New: {alpha_value}")
                            all_sample_info[image_id]['mixed_file'] = filename  # Update in case of duplicates

        # Process pure source files
        for filename in os.listdir(pure_source_dir):
            if filename.endswith('.tif'):
                match = self.file_pattern.search(filename)
                if match:
                    image_id = match.group(1)
                    alpha_value = float(match.group(2))
                    file_type = match.group(3)  # 'mixed' or 'source'

                    if file_type == 'source':
                        if image_id not in all_sample_info:
                            all_sample_info[image_id] = {'alpha': alpha_value, 'source_file': filename}
                        else:
                            # Basic consistency check: alpha should be same for the same ID
                            if all_sample_info[image_id]['alpha'] != alpha_value:
                                print(
                                    f"Warning: Alpha mismatch for ID {image_id} (source). Existing: {all_sample_info[image_id]['alpha']}, New: {alpha_value}")
                            all_sample_info[image_id]['source_file'] = filename  # Update

        self.samples = []
        for image_id, info in all_sample_info.items():
            if 'mixed_file' in info and 'source_file' in info:
                self.samples.append({
                    'image_id': image_id,
                    'mixed_channel_file': info['mixed_file'],
                    'pure_source_file': info['source_file'],
                    'scalar_label': info['alpha']
                })
            else:
                # print(f"Warning: Skipping ID {image_id} due to missing mixed or source file.")
                pass  # Already printing warnings if file doesn't match regex.

        if not self.samples:
            raise ValueError(
                "No matching samples found. Ensure filenames adhere to 'image_ID_alpha_VALUE_(mixed|source).tif' pattern and corresponding mixed/source files exist.")

        # Sort samples by image_id for consistent order (important for splitting)
        self.samples = sorted(self.samples, key=lambda x: x['image_id'])
        print(f"Found {len(self.samples)} matching samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        mixed_channel_path = os.path.join(self.mixed_channel_dir, sample_info['mixed_channel_file'])
        pure_source_path = os.path.join(self.pure_source_dir, sample_info['pure_source_file'])
        scalar_label = sample_info['scalar_label']

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)

        if self.transform:
            mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, scalar_label)
        else:
            mixed_tensor = torch.from_numpy(mixed_image_np).unsqueeze(0)
            source_tensor = torch.from_numpy(source_image_np).unsqueeze(0)
            label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


class SplitCrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, transform, split_samples):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.transform = transform
        self.split_samples = split_samples  # This is already the filtered list of sample dicts

        if not self.split_samples:
            raise ValueError("SplitCrosstalkDataset received no samples.")
        print(f"SplitCrosstalkDataset created with {len(self.split_samples)} samples.")

    def __len__(self):
        return len(self.split_samples)

    def __getitem__(self, idx_in_split):
        sample_info = self.split_samples[idx_in_split]

        mixed_channel_path = os.path.join(self.mixed_channel_dir, sample_info['mixed_channel_file'])
        pure_source_path = os.path.join(self.pure_source_dir, sample_info['pure_source_file'])
        scalar_label = sample_info['scalar_label']

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)

        mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, scalar_label)
        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


# --- End of Custom Dataset Classes ---


# --- Transform Functions (no changes needed) ---
def get_train_transforms(target_size):
    def train_transforms_fn(mixed_np, source_np, scalar_label):
        mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
        source_tensor = torch.from_numpy(source_np).unsqueeze(0)
        label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

        if torch.rand(1) < 0.5:
            mixed_tensor = TF.hflip(mixed_tensor)
            source_tensor = TF.hflip(source_tensor)

        if torch.rand(1) < 0.5:
            mixed_tensor = TF.vflip(mixed_tensor)
            source_tensor = TF.vflip(source_tensor)

        return mixed_tensor, source_tensor, label_tensor

    return train_transforms_fn


def get_val_test_transforms(target_size):
    def val_test_transforms_fn(mixed_np, source_np, scalar_label):
        mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
        source_tensor = torch.from_numpy(source_np).unsqueeze(0)
        label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

        return mixed_tensor, source_tensor, label_tensor

    return val_test_transforms_fn


# --- End of Transform Functions ---


# --- Training Function (no changes needed) ---
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    log_file_path = "training_log_regression.csv"
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

                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)

            epoch_val_loss = val_running_loss / len(val_dataloader.dataset)
            print(f"Epoch {epoch + 1} Validation Loss: {epoch_val_loss:.6f}")

            log_writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss])
            log_file.flush()

    print("Training complete. Losses logged to training_log_regression.csv.")


# --- End of Training Function ---


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mixed_channel_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/bleed"
    pure_source_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/source"

    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    TARGET_IMAGE_SIZE = (256, 256)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    if not (abs(TRAIN_RATIO + VAL_RATIO) < 1.0):
        print("Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0.")

    model = RegressionModel()

    print("\nCreating dataset instances for initial file listing...")
    try:
        # No label_mapping_file needed, as labels are derived from filenames
        temp_dataset = CrosstalkDataset(
            mixed_channel_data_dir, pure_source_data_dir,
            transform=get_val_test_transforms(TARGET_IMAGE_SIZE)
        )
        print(f"Total samples found in directories: {len(temp_dataset)}")
    except Exception as e:
        print(f"Error initializing temporary dataset: {e}")
        exit()

    print("\nSplitting data using filename lists for correct augmentation application...")
    all_samples = temp_dataset.samples  # This now holds the list of dictionaries with all sample info
    total_samples = len(all_samples)
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(total_samples).tolist()

    train_size = int(TRAIN_RATIO * total_samples)
    val_size = int(VAL_RATIO * total_samples)
    test_size = total_samples - train_size - val_size

    train_samples = [all_samples[i] for i in shuffled_indices[0:train_size]]
    val_samples = [all_samples[i] for i in shuffled_indices[train_size: train_size + val_size]]
    test_samples = [all_samples[i] for i in shuffled_indices[train_size + val_size:]]

    print(f"Split sizes: Train = {len(train_samples)}, Validation = {len(val_samples)}, Test = {len(test_samples)}")

    train_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=get_train_transforms(TARGET_IMAGE_SIZE),
        split_samples=train_samples
    )

    val_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=get_val_test_transforms(TARGET_IMAGE_SIZE),
        split_samples=val_samples
    )

    test_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=get_val_test_transforms(TARGET_IMAGE_SIZE),
        split_samples=test_samples
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

    model_save_path = "crosstalk_regression_model_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = RegressionModel()
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    loaded_model.eval()
    loaded_model.to(device)

    test_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Test Set Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = loaded_model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)

    final_test_loss = test_running_loss / len(test_dataloader.dataset)
    print(f"\nFinal Test Loss: {final_test_loss:.6f}")
    print("Test set evaluation complete.")
