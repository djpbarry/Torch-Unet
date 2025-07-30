import csv
import os
import random
import re  # Import regex for pattern matching

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from two_branch_regression import TwoBranchRegressionModel


# --- Custom Dataset Classes (MODIFIED to use (image_id, alpha_value) as key) ---
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, transform=None, max_samples=None):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.transform = transform

        # Regex to extract the unique ID, the alpha value, and the file type
        # This pattern works for both '_mixed.tif' and '_source.tif' files
        self.file_pattern = re.compile(r'image_(\d+)_alpha_(\d+\.?\d*)_(mixed|source)\.tif')

        # Store mappings: {(image_id, alpha_value_str): {mixed_file: path, source_file: path}}
        # Using alpha_value_str (from regex group) as part of the key to avoid float comparison issues
        all_sample_info = {}

        # Helper to process files from a directory
        def process_files_in_dir(directory, file_type_key):
            for filename in os.listdir(directory):
                if filename.endswith('.tif'):
                    match = self.file_pattern.search(filename)
                    if match:
                        image_id = match.group(1)
                        alpha_value_str = match.group(2)  # Keep as string for the key
                        file_actual_type = match.group(3)

                        # Only process files that match the expected type for this directory
                        if file_actual_type == file_type_key:
                            compound_key = (image_id, alpha_value_str)
                            if compound_key not in all_sample_info:
                                all_sample_info[compound_key] = {}
                            all_sample_info[compound_key][f'{file_type_key}_file'] = filename

        # Process mixed channel files
        process_files_in_dir(mixed_channel_dir, 'mixed')
        # Process pure source files
        process_files_in_dir(pure_source_dir, 'source')

        self.samples = []
        for (image_id, alpha_value_str), info in all_sample_info.items():
            if 'mixed_file' in info and 'source_file' in info:
                self.samples.append({
                    'image_id': image_id,  # Can keep for debugging/sorting
                    'scalar_label': float(alpha_value_str),  # Convert to float for label
                    'mixed_channel_file': info['mixed_file'],
                    'pure_source_file': info['source_file']
                })

        if not self.samples:
            raise ValueError(
                "No matching samples found. Ensure filenames adhere to 'image_ID_alpha_VALUE_(mixed|source).tif' "
                "pattern and corresponding mixed/source files exist for each (ID, Alpha) pair.")

        # Sort samples for consistent order (important for splitting)
        # Sorting by image_id and then scalar_label to ensure stable order
        self.samples.sort(key=lambda x: (x['image_id'], x['scalar_label']))

        if max_samples:
            self.samples = self.samples[:max_samples]
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

    img_h, img_w = mixed_tensor.shape[-2:]

    # Example for rotation (needs to be applied to both identically)
    angle = random.uniform(-15, 15)  # Get random angle

    # Random translation (e.g., up to 10% of width/height)
    # translate = (horizontal_shift_percentage, vertical_shift_percentage)
    translate_x = random.uniform(-0.1, 0.1) * img_w
    translate_y = random.uniform(-0.1, 0.1) * img_h
    translate = [translate_x, translate_y]

    # Apply the generated affine transform to both tensors
    # TF.affine(img, angle, translate, scale, shear, interpolation, fill)
    # For single-channel images, interpolation=TF.InterpolationMode.BILINEAR is good.
    # fill=0.0 (or pixel mean) for areas outside the image after transform.
    mixed_tensor = TF.affine(
        mixed_tensor,
        angle=angle,
        translate=translate,
        scale=1.0,
        shear=[0.0],
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0.0]  # Fill value for pixels outside original image
    )
    source_tensor = TF.affine(
        source_tensor,
        angle=angle,
        translate=translate,
        scale=1.0,
        shear=[0.0],
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=[0.0]
    )

    return mixed_tensor, source_tensor, label_tensor


def val_test_transforms_fn(mixed_np, source_np, scalar_label):
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    return mixed_tensor, source_tensor, label_tensor


# --- End of Transform Functions ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
                log_csv_path='training_log.csv'):
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)

    # Prepare the CSV log file
    with open(log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])  # write header

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Log to CSV
            writer.writerow([epoch + 1, train_loss, val_loss])

    return train_losses, val_losses


# --- End of Training Function ---


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mixed_channel_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data_3/bleed"
    pure_source_data_dir = "/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data_3/source"

    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    TARGET_IMAGE_SIZE = (256, 256)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    if not (abs(TRAIN_RATIO + VAL_RATIO) < 1.0):
        print("Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0.")

    model = TwoBranchRegressionModel(initial_filters_per_branch=32, input_image_size=TARGET_IMAGE_SIZE)

    print("\nCreating dataset instances for initial file listing...")
    try:
        # No label_mapping_file needed, as labels are derived from filenames
        temp_dataset = CrosstalkDataset(
            mixed_channel_data_dir, pure_source_data_dir,
            transform=val_test_transforms_fn
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
        transform=train_transforms_fn,
        split_samples=train_samples
    )

    val_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=val_test_transforms_fn,
        split_samples=val_samples
    )

    test_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=val_test_transforms_fn,
        split_samples=test_samples
    )

    train_dataloader = DataLoader(
        train_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset_final,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=int(os.getenv('SLURM_CPUS_PER_TASK', default=1)),
        pin_memory=True,
        drop_last=True
    )

    print("Dataloaders created for training, validation, and testing.")

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\nStarting training with validation...")
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS,
                                           device)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig('Training_Loss.pdf')

    print("Training finished!")

    model_save_path = "crosstalk_regression_model_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = TwoBranchRegressionModel(initial_filters_per_branch=32, input_image_size=TARGET_IMAGE_SIZE)
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    loaded_model.eval()
    loaded_model.to(device)

    test_running_loss = 0.0
    # List to store actual vs. predicted values
    predictions_data = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_dataloader, desc="Test Set Evaluation")):
            inputs = inputs.to(device)
            labels = labels.to(device)  # Shape: (batch_size, 1)

            outputs = loaded_model(inputs)  # Shape: (batch_size, 1)

            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * inputs.size(0)

            # Store actual and predicted values
            # Move tensors to CPU and convert to numpy arrays, then flatten to 1D list
            actual_labels = labels.cpu().numpy().flatten()
            predicted_labels = outputs.cpu().numpy().flatten()

            # For each sample in the current batch, append its actual and predicted value
            for j in range(len(actual_labels)):
                predictions_data.append({
                    'Actual_Label': actual_labels[j],
                    'Predicted_Label': predicted_labels[j]
                })

    final_test_loss = test_running_loss / len(test_dataloader.dataset)
    print(f"\nFinal Test Loss: {final_test_loss:.6f}")
    print("Test set evaluation complete.")

    # --- Save predictions to CSV ---
    output_csv_path = "test_predictions.csv"
    with open(output_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Actual_Label', 'Predicted_Label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(predictions_data)

    print(f"Test predictions saved to {output_csv_path}")
