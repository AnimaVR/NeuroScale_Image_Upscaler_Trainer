import os
import shutil
import numpy as np
import cv2
import torch
from utils.video.mov_extraction import find_files, extract_frames

def organize_videos(root_dir):
    """
    Checks for video files (e.g. .mov, .mp4) in root_dir that are not inside a folder.
    For each video found, creates a folder (named after the video file's base name),
    and moves the video file into that folder.
    """
    video_extensions = (".mov", ".mp4")
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isfile(entry_path) and entry.lower().endswith(video_extensions):
            base_name = os.path.splitext(entry)[0]
            new_folder = os.path.join(root_dir, base_name)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, entry)
            shutil.move(entry_path, new_path)
            print(f"Moved video '{entry}' to folder '{new_folder}'")

def load_data(root_dir, processed_folders):
    """
    Loads data from multiple folders containing videos and/or images.
    Returns a list of pairs: (small_frames, large_frames) for training.
    Each is a tensor of shape (total_scanlines, 768).
    """
    # First, organize any loose video files into their own folders.
    organize_videos(root_dir)

    examples = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and folder not in processed_folders:
            small_frames, large_frames = process_folder(folder_path)
            if small_frames is not None and large_frames is not None:
                examples.append((small_frames, large_frames))
                processed_folders.add(folder)
    return examples

def process_folder(folder_path):
    """
    Processes a folder by extracting frames from a video or using images found,
    and preparing dataset tensors.
    """
    mov_path, mp4_path, small_frames_dir, large_frames_dir = find_files(folder_path)
    video_path = mov_path or mp4_path  # Use whichever video is found

    if video_path:
        # If a video file exists, extract frames if needed.
        existing_small = sorted(os.listdir(small_frames_dir))
        existing_large = sorted(os.listdir(large_frames_dir))
        if existing_small and existing_large and len(existing_small) == len(existing_large):
            print(f"Using existing extracted frames for {folder_path}")
        else:
            extract_frames(video_path, small_frames_dir, large_frames_dir)
    else:
        # No video file found; check for image files.
        print(f"No video found in {folder_path}. Looking for images.")
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
        if not image_files:
            print(f"No video or images found in {folder_path}")
            return None, None

        # Ensure the frames directories exist.
        os.makedirs(small_frames_dir, exist_ok=True)
        os.makedirs(large_frames_dir, exist_ok=True)

        # Process each image as a frame.
        for idx, image_file in enumerate(sorted(image_files)):
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Skipping image {image_file} due to loading error.")
                continue
            # Use a cover strategy to avoid distortion (if desired, you can use a helper like resize_cover)
            small_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            large_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            # Save processed images to the frames directories.
            small_frame_path = os.path.join(small_frames_dir, f"frame_{idx:04d}.png")
            large_frame_path = os.path.join(large_frames_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(small_frame_path, small_img)
            cv2.imwrite(large_frame_path, large_img)

    # Process extracted frames (whether from video or images) into tensors.
    return collect_features(small_frames_dir, large_frames_dir)

def collect_features(small_frames_dir, large_frames_dir):
    """
    Loads extracted frames in color, normalizes pixel values, and converts them into tokens.
    Each frame is resized to 256x256.
    Each row (scanline) is flattened from (256, 3) to a vector of length 768.
    Returns:
        tuple: (small_sequences_tensor, large_sequences_tensor) of shape (TotalScanlines, 768)
    """
    small_sequences, large_sequences = [], []
    # Get sorted frame filenames to maintain order.
    frame_filenames = sorted(os.listdir(small_frames_dir))

    for frame_name in frame_filenames:
        small_path = os.path.join(small_frames_dir, frame_name)
        large_path = os.path.join(large_frames_dir, frame_name)

        # Load images in color.
        small_img = cv2.imread(small_path, cv2.IMREAD_COLOR)
        large_img = cv2.imread(large_path, cv2.IMREAD_COLOR)

        if small_img is None or large_img is None:
            print(f"Skipping {frame_name} due to loading error.")
            continue

        # Convert from BGR to RGB.
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)

        # Resize images to 256x256.
        small_resized = cv2.resize(small_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        large_resized = cv2.resize(large_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Normalize pixel values to [0,1].
        small_norm = small_resized.astype(np.float32) / 255.0
        large_norm = large_resized.astype(np.float32) / 255.0

        # Flatten each row (256,3) -> (768,).
        flat_small = small_norm.reshape(256, -1)
        flat_large = large_norm.reshape(256, -1)

        # Append each row as a separate token.
        for row in flat_small:
            small_sequences.append(row)
        for row in flat_large:
            large_sequences.append(row * 20.0)

    # Convert lists to tensors of shape (TotalScanlines, 768).
    small_sequences_tensor = torch.tensor(np.vstack(small_sequences), dtype=torch.float32)
    large_sequences_tensor = torch.tensor(np.vstack(large_sequences), dtype=torch.float32)

    return small_sequences_tensor, large_sequences_tensor
