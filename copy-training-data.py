import os
import shutil
import random
from pathlib import Path


def get_random_indices(source_dir, num_files=100):
    """
    Get random file indices from a source directory.

    Args:
        source_dir: Path to source directory
        num_files: Number of files to select (default: 100)

    Returns:
        tuple: (list of selected indices, list of all files)
    """
    source = Path(source_dir)

    if not source.exists():
        raise ValueError(f"Source directory does not exist: {source}")

    if not source.is_dir():
        raise ValueError(f"Source path is not a directory: {source}")

    # Get all files from source directory
    all_files = [f for f in source.iterdir() if f.is_file()]

    if not all_files:
        raise ValueError("No files found in source directory.")

    # Determine how many files to select
    files_to_select = min(num_files, len(all_files))

    if files_to_select < num_files:
        print(f"Warning: Only {len(all_files)} files available, selecting all of them.")

    # Generate random indices
    indices = random.sample(range(len(all_files)), files_to_select)

    return indices, all_files


def copy_files_by_indices(source_dir, target_dir, indices):
    """
    Copy files from source to target directory using specific indices.

    Args:
        source_dir: Path to source directory
        target_dir: Path to target directory
        indices: List of file indices to copy
    """
    source = Path(source_dir)
    target = Path(target_dir)

    if not source.exists():
        raise ValueError(f"Source directory does not exist: {source}")

    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)

    # Get all files
    all_files = [f for f in source.iterdir() if f.is_file()]

    if not all_files:
        print(f"No files found in {source_dir}")
        return

    # Copy files at specified indices
    copied_count = 0
    for idx in indices:
        if idx >= len(all_files):
            print(f"Warning: Index {idx} out of range, skipping.")
            continue

        file_path = all_files[idx]
        try:
            dest_path = target / file_path.name
            shutil.copy2(file_path, dest_path)
            copied_count += 1
            print(f"Copied: {file_path.name}")
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")

    print(f"Successfully copied {copied_count} files from {source_dir}\n")


if __name__ == "__main__":
    # Define your directories
    SOURCE_DIR_1 = "Z:/working/barryd/IDR/crosstalk_training_data/source"
    TARGET_DIR_1 = "./Training_Data/Source"

    SOURCE_DIR_2 = "Z:/working/barryd/IDR/crosstalk_training_data/bleed"
    TARGET_DIR_2 = "./Training_Data/Mixed"

    NUM_FILES = 100

    # Get random indices from first directory
    print(f"Selecting {NUM_FILES} random files from first directory...")
    indices, files = get_random_indices(SOURCE_DIR_1, NUM_FILES)
    print(f"Selected {len(indices)} file indices\n")

    # Copy from first directory
    print("Copying from first directory...")
    copy_files_by_indices(SOURCE_DIR_1, TARGET_DIR_1, indices)

    # Copy same indices from second directory
    print("Copying same file indices from second directory...")
    copy_files_by_indices(SOURCE_DIR_2, TARGET_DIR_2, indices)