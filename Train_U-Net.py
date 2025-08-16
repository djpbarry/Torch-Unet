import argparse
import csv
import datetime
import os
import re  # Import regex for pattern matching

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from regression_model import *

# from datetime import datetime

TARGET_IMAGE_SIZE = (256, 256)


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


def train_transforms_fn(mixed_np, source_np, scalar_label):
    # Convert numpy arrays to PyTorch tensors and add channel dimension
    # Ensure pixel values are normalized to [0, 1] if not already.
    # IMPORTANT: Apply normalization here by dividing by 255.0 (or max pixel value)
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    # Horizontal Flip
    if torch.rand(1) < 0.5:
        mixed_tensor = TF.hflip(mixed_tensor)
        source_tensor = TF.hflip(source_tensor)

    # Vertical Flip
    if torch.rand(1) < 0.5:
        mixed_tensor = TF.vflip(mixed_tensor)
        source_tensor = TF.vflip(source_tensor)

    # img_h, img_w = mixed_tensor.shape[-2:]
    #
    # # Affine Transform Parameters:
    # degrees = random.uniform(-15, 15)
    # translate_x = random.uniform(-0.1, 0.1) * img_w
    # translate_y = random.uniform(-0.1, 0.1) * img_h
    # translate = [translate_x, translate_y]
    # scale = 1.0
    # shear = [0.0]
    # fill = [0.0]
    #
    # # Apply the generated affine transform to both tensors
    # mixed_tensor = TF.affine(
    #     mixed_tensor,
    #     angle=degrees,
    #     translate=translate,
    #     scale=scale,
    #     shear=shear,
    #     interpolation=TF.InterpolationMode.BILINEAR,
    #     fill=fill
    # )
    # source_tensor = TF.affine(
    #     source_tensor,
    #     angle=degrees,
    #     translate=translate,
    #     scale=scale,
    #     shear=shear,
    #     interpolation=TF.InterpolationMode.BILINEAR,
    #     fill=fill
    # )

    # # 3. Add Gaussian Noise (applied identically to both images)
    # # Adjust mean and std based on your image intensity range (now 0-1)
    # noise_mean = 0.0
    # noise_std = random.uniform(0.01, 0.05)  # Experiment with this range (e.g., 1-5% of max intensity)
    # gaussian_noise = torch.randn(mixed_tensor.shape) * noise_std + noise_mean
    #
    # mixed_tensor = mixed_tensor + gaussian_noise
    # source_tensor = source_tensor + gaussian_noise
    #
    # # Clip values to ensure they remain within [0, 1] after adding noise
    # mixed_tensor = torch.clamp(mixed_tensor, 0.0, 1.0)
    # source_tensor = torch.clamp(source_tensor, 0.0, 1.0)

    # 4. Add Random Erasing (applied identically to both images for consistency)
    # Generate random parameters for erasing once
    # if random.random() < 0.5:  # Probability of applying Random Erasing
    #     # Get random parameters for erasing block
    #     # i, j: top-left corner coordinates of the erased block
    #     # h, w: height and width of the erased block
    #     # v: value to fill the erased block with
    #     area_ratio = random.uniform(0.02, 0.1)  # Erase 2% to 10% of the image area
    #     aspect_ratio = random.uniform(0.3, 3.3)  # Aspect ratio of the erased block
    #
    #     # Calculate h and w from area_ratio and aspect_ratio
    #     h = int(np.sqrt(area_ratio * img_h * img_w / aspect_ratio))
    #     w = int(np.sqrt(area_ratio * img_h * img_w * aspect_ratio))
    #
    #     # Ensure h and w are not zero
    #     h = max(1, h)
    #     w = max(1, w)
    #
    #     i = random.randint(0, img_h - h)
    #     j = random.randint(0, img_w - w)
    #
    #     mixed_tensor = TF.erase(
    #         mixed_tensor, i, j, h, w, v=0.0
    #     )
    #     source_tensor = TF.erase(
    #         source_tensor, i, j, h, w, v=0.0
    #     )

    return mixed_tensor, source_tensor, label_tensor


def val_test_transforms_fn(mixed_np, source_np, scalar_label):
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    return mixed_tensor, source_tensor, label_tensor


# --- End of Transform Functions ---

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_dir):
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)

    # Create the timestamped log filename
    timestamped_log_file = os.path.join(output_dir,
                                        f"training_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{batch_size}_{learning_rate}.csv")

    # Prepare the CSV log file
    with open(timestamped_log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Learning Rate', learning_rate])
        writer.writerow(['Batch Size', batch_size])
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
    parser = argparse.ArgumentParser(description="Script for training with various parameters.")

    parser.add_argument("-m", "--mixed_channel_data_dir", type=str,
                        default="/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/bleed",
                        help="Directory for mixed channel data")
    parser.add_argument("-s", "--pure_source_data_dir", type=str,
                        default="/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/source",
                        help="Directory for pure source data")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("-n", "--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("-t", "--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("-v", "--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("-j", "--cpu_jobs", type=int, default=1, help="Number of CPUs to use")

    args = parser.parse_args()

    mixed_channel_data_dir = args.mixed_channel_data_dir
    pure_source_data_dir = args.pure_source_data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    ncpus = args.cpu_jobs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not (abs(train_ratio + val_ratio) < 1.0):
        print("Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0.")

    model = AdvancedRegressionModel(initial_filters=128, num_conv_blocks=6)
    print(f'Using {ncpus} cpu workers.')

    # --- Create a unique output directory for this run ---
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Include batch size and learning rate in the folder name for easy identification
    output_dir_name = f"training_run_{current_time_str}_B{batch_size}_LR{learning_rate}"
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Saving all outputs to: {output_dir_name}")

    # --- Save Model Architecture Summary ---
    model_summary_path = os.path.join(output_dir_name, "model_architecture.txt")
    with open(model_summary_path, "w") as f:
        f.write(str(model))  # This prints the __repr__ of the model
    print(f"Model architecture summary saved to {model_summary_path}")

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
    torch.manual_seed(43)
    shuffled_indices = torch.randperm(total_samples).tolist()

    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    print("Dataloaders created for training, validation, and testing.")

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    print("\nStarting training with validation...")
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs,
                                           device, output_dir_name)

    print("Training finished!")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(output_dir_name,
                                   f"crosstalk_regression_model_trained_{current_time}_{batch_size}_{learning_rate}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    # --- Plot Training and Validation Losses ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=0, top=0.02)
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir_name,
                                  f"training_validation_loss_{current_time}_{batch_size}_{learning_rate}.png")
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    plt.close()  # Close the plot to free memory

    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = AdvancedRegressionModel(initial_filters=128, num_conv_blocks=6)
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
    output_csv_path = os.path.join(output_dir_name, f"test_predictions_{current_time}_{batch_size}_{learning_rate}.csv")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        # Use a regular csv.writer to write the parameters first
        writer = csv.writer(csv_file)
        writer.writerow(['Learning Rate', learning_rate])
        writer.writerow(['Batch Size', batch_size])

        # Then, define the fieldnames for the DictWriter
        fieldnames = ['Actual_Label', 'Predicted_Label']
        dict_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        dict_writer.writeheader()
        dict_writer.writerows(predictions_data)

    print(f"Test predictions saved to {output_csv_path}")

    # --- Plot Test Results (Actual vs. Predicted) ---
    actual_labels_all = [d['Actual_Label'] for d in predictions_data]
    predicted_labels_all = [d['Predicted_Label'] for d in predictions_data]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual_labels_all, predicted_labels_all, alpha=0.6, s=10)
    plt.plot([min(actual_labels_all), max(actual_labels_all)],
             [min(actual_labels_all), max(actual_labels_all)],
             '--r', label='Ideal Prediction (y=x)')  # Plot a y=x line
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.title("Test Set: Actual vs. Predicted Labels")
    plt.legend()
    test_plot_path = os.path.join(output_dir_name,
                                  f"test_predictions_plot_{current_time}_{batch_size}_{learning_rate}.png")
    plt.savefig(test_plot_path)
    print(f"Test predictions plot saved to {test_plot_path}")
    plt.close()  # Close the plot to free memory

    train_predictions_data = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader, desc="Train Set Evaluation")):
            inputs = inputs.to(device)
            labels = labels.to(device)  # Shape: (batch_size, 1)

            outputs = loaded_model(inputs)  # Shape: (batch_size, 1)

            # Store actual and predicted values
            # Move tensors to CPU and convert to numpy arrays, then flatten to 1D list
            actual_labels = labels.cpu().numpy().flatten()
            predicted_labels = outputs.cpu().numpy().flatten()

            # For each sample in the current batch, append its actual and predicted value
            for j in range(len(actual_labels)):
                train_predictions_data.append({
                    'Actual_Label': actual_labels[j],
                    'Predicted_Label': predicted_labels[j]
                })

    actual_labels_all = [d['Actual_Label'] for d in train_predictions_data]
    predicted_labels_all = [d['Predicted_Label'] for d in train_predictions_data]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual_labels_all, predicted_labels_all, alpha=0.6, s=10)
    plt.plot([min(actual_labels_all), max(actual_labels_all)],
             [min(actual_labels_all), max(actual_labels_all)],
             '--r', label='Ideal Prediction (y=x)')  # Plot a y=x line
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.title("Train Set: Actual vs. Predicted Labels")
    plt.legend()
    test_plot_path = os.path.join(output_dir_name,
                                  f"train_predictions_plot_{current_time}_{batch_size}_{learning_rate}.png")
    plt.savefig(test_plot_path)
    print(f"Train predictions plot saved to {test_plot_path}")
    plt.close()  # Close the plot to free memory

    val_predictions_data = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_dataloader, desc="Val Set Evaluation")):
            inputs = inputs.to(device)
            labels = labels.to(device)  # Shape: (batch_size, 1)

            outputs = loaded_model(inputs)  # Shape: (batch_size, 1)

            # Store actual and predicted values
            # Move tensors to CPU and convert to numpy arrays, then flatten to 1D list
            actual_labels = labels.cpu().numpy().flatten()
            predicted_labels = outputs.cpu().numpy().flatten()

            # For each sample in the current batch, append its actual and predicted value
            for j in range(len(actual_labels)):
                val_predictions_data.append({
                    'Actual_Label': actual_labels[j],
                    'Predicted_Label': predicted_labels[j]
                })

    actual_labels_all = [d['Actual_Label'] for d in val_predictions_data]
    predicted_labels_all = [d['Predicted_Label'] for d in val_predictions_data]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual_labels_all, predicted_labels_all, alpha=0.6, s=10)
    plt.plot([min(actual_labels_all), max(actual_labels_all)],
             [min(actual_labels_all), max(actual_labels_all)],
             '--r', label='Ideal Prediction (y=x)')  # Plot a y=x line
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.title("Val Set: Actual vs. Predicted Labels")
    plt.legend()
    test_plot_path = os.path.join(output_dir_name,
                                  f"val_predictions_plot_{current_time}_{batch_size}_{learning_rate}.png")
    plt.savefig(test_plot_path)
    print(f"Val predictions plot saved to {test_plot_path}")
    plt.close()  # Close the plot to free memory
