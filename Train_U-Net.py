import os

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# --- 1. Custom PyTorch Dataset for your Crosstalk Data ---
class CrosstalkDataset(Dataset):
    """
    Custom PyTorch Dataset for loading (mixed_channel, pure_source_channel) as input
    and crosstalk ground truth map as label.
    """

    def __init__(self, mixed_channel_dir, pure_source_dir, label_dir, target_size):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.label_dir = label_dir
        self.target_size = target_size

        self.mixed_channel_filenames = sorted([f for f in os.listdir(mixed_channel_dir) if f.endswith('.tif')])
        self.pure_source_filenames = sorted([f for f in os.listdir(pure_source_dir) if f.endswith('.tif')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        if not (len(self.mixed_channel_filenames) == len(self.pure_source_filenames) == len(self.label_filenames)):
            raise ValueError(
                "Number of mixed channel, pure source, and label images must be the same."
            )

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

        mixed_image = iio.imread(mixed_channel_path).astype(np.float32)
        source_image = iio.imread(pure_source_path).astype(np.float32)
        label_image = iio.imread(label_path).astype(np.float32)

        current_h, current_w = mixed_image.shape
        if (current_h, current_w) != self.target_size:
            resize_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(self.target_size),
                T.ToTensor()
            ])

            mixed_tensor = resize_transform(mixed_image)
            source_tensor = resize_transform(source_image)
            label_tensor = resize_transform(label_image)

        else:
            mixed_tensor = torch.from_numpy(mixed_image).unsqueeze(0)
            source_tensor = torch.from_numpy(source_image).unsqueeze(0)
            label_tensor = torch.from_numpy(label_image).unsqueeze(0)

        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)

        return input_tensor, label_tensor


# --- 2. Training Function ---
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    scaler = amp.GradScaler()  # For mixed precision training

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(inputs)
                if outputs.shape != labels.shape:
                    outputs = nn.functional.interpolate(outputs, size=labels.shape[2:], mode='bilinear',
                                                        align_corners=False)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch + 1} Train Loss: {epoch_train_loss:.6f}")

        # Validation phase
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


# --- 3. Main Execution Block (MODIFIED for train/val/test split) ---
if __name__ == "__main__":
    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data directories ---
    mixed_channel_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/bleed"
    pure_source_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/source"
    label_data_dir = "/nemo/stp/lm/working/barryd/IDR/Cross-Talk-Training-Data/output/ground_truth"

    # --- Hyperparameters ---
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 500
    U_NET_IN_CHANNELS = 2
    U_NET_OUT_CHANNELS = 1
    TARGET_IMAGE_SIZE = (256, 256)  # Adjust this value based on your image sizes!

    # --- Data split ratios (NEW: added TEST_RATIO) ---
    TRAIN_RATIO = 0.7  # 70% for training
    VAL_RATIO = 0.15  # 15% for validation
    TEST_RATIO = 0.15  # 15% for testing
    # Ensure TRAIN_RATIO + VAL_RATIO + TEST_RATIO sums to 1.0 (or close to it due to float precision)

    if not (TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0):
        print(
            "Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0. Data might be left out or sizes might be adjusted.")

    # --- Load U-Net Model from PyTorch Hub ---
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

    # --- Create full dataset ---
    print("\nCreating full dataset...")
    try:
        full_dataset = CrosstalkDataset(mixed_channel_data_dir, pure_source_data_dir, label_data_dir,
                                        target_size=TARGET_IMAGE_SIZE)
        print(f"Total samples loaded: {len(full_dataset)}")
    except Exception as e:
        print(f"Error creating full dataset: {e}")
        print(
            "Please ensure the data directories "
            f"('{mixed_channel_data_dir}', '{pure_source_data_dir}', and '{label_data_dir}') "
            "exist and contain matching .tif files."
        )
        exit()

    # --- Split dataset into training, validation, and test sets ---
    total_size = len(full_dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size  # The remainder goes to test
    print(f"Splitting data: Train samples = {train_size}, Validation samples = {val_size}, Test samples = {test_size}")

    # Ensure reproducibility of the split
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # --- Create dataloaders for training, validation, and testing ---
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=os.cpu_count() // 2 or 1)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=os.cpu_count() // 2 or 1)  # Test set not shuffled

    print("Dataloaders created for training, validation, and testing.")

    # --- Initialize loss function and optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training with validation...")
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS, device)
    print("Training finished!")

    # --- Save the trained model's weights ---
    model_save_path = "crosstalk_detection_unet_2input_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    # --- Final Model Evaluation on the Test Set (NEW) ---
    print("\n--- Evaluating Model on Test Set ---")
    loaded_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                  in_channels=U_NET_IN_CHANNELS, out_channels=U_NET_OUT_CHANNELS,
                                  init_features=32, pretrained=False)
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))
    loaded_model.eval()  # Set model to evaluation mode
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

    # --- Inference Example with the Trained Model (using a sample from test_dataset) ---
    print("\n--- Running Inference Example with Trained Model (using a test sample) ---")
    if len(test_dataset) > 0:
        # Get the first sample from the test dataset for demonstration
        sample_input_tensor, sample_label_tensor = test_dataset[0]

        # Use the already loaded and evaluated model
        with torch.no_grad():
            input_for_inference = sample_input_tensor.unsqueeze(0).to(device)
            predicted_crosstalk_map = loaded_model(input_for_inference).squeeze(0).cpu().numpy()

        display_mixed_input = sample_input_tensor[0].cpu().numpy()
        display_source_input = sample_input_tensor[1].cpu().numpy()
        display_label = sample_label_tensor.squeeze().cpu().numpy()
        display_predicted = predicted_crosstalk_map.squeeze()

        display_predicted = np.clip(display_predicted, 0, display_label.max())

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(display_mixed_input, cmap='magma')
        axes[0].set_title('Input: Mixed Channel')
        axes[0].axis('off')

        axes[1].imshow(display_source_input, cmap='viridis')
        axes[1].set_title('Input: Pure Source Channel')
        axes[1].axis('off')

        axes[2].imshow(display_label, cmap='gray')
        axes[2].set_title('True Crosstalk Map (Ground Truth)')
        axes[2].axis('off')

        axes[3].imshow(display_predicted, cmap='gray')
        axes[3].set_title('Predicted Crosstalk Map')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("No samples in test dataset to perform inference example.")
