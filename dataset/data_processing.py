import os
import numpy as np
import cv2
import torch
from utils.video.mov_extraction import find_files, extract_frames

def resize_cover(image, target_size, interpolation=cv2.INTER_CUBIC):
    """
    Resizes an image to the target size using a "cover" strategy.
    The image is scaled to fill the target size while preserving its aspect ratio,
    then center-cropped to exactly match target_size.
    
    :param image: Input image (H x W x C).
    :param target_size: Desired size as (width, height).
    :param interpolation: Interpolation method.
    :return: Resized and cropped image.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    # Determine scale factor: the larger of the width or height ratios.
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Resize the image with the computed scale.
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    # Compute center crop coordinates.
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    cropped = resized[y:y+target_h, x:x+target_w]
    return cropped

def load_data(root_dir, processed_folders):
    """
    Loads data from multiple folders containing videos and/or images.
    Returns a list of pairs: (small_frames, large_frames) for training.
    Each is a tensor of shape (total_scanlines, 768).
    """
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
            # Use resize_cover to avoid distortion.
            small_img = resize_cover(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            large_img = resize_cover(img, (256, 256), interpolation=cv2.INTER_CUBIC)
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
    Each frame is resized to 256x256 using the cover method.
    Each row (scanline) is flattened from (256,3) to a vector of length 768.
    
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

        # Use resize_cover to ensure the images fill 256x256 without distortion.
        small_resized = resize_cover(small_img, (256, 256), interpolation=cv2.INTER_NEAREST)
        large_resized = resize_cover(large_img, (256, 256), interpolation=cv2.INTER_CUBIC)

        # Normalize pixel values to [0,1].
        small_norm = small_resized.astype(np.float32) / 255.0
        large_norm = large_resized.astype(np.float32) / 255.0

        # Flatten each row (256,3) -> (768,).
        flat_small = small_norm.reshape(256, -1)  # shape: (256, 768)
        flat_large = large_norm.reshape(256, -1)    # shape: (256, 768)

        # Append each row as a separate token.
        for row in flat_small:
            small_sequences.append(row)
        for row in flat_large:
            large_sequences.append(row * 10.0)

    # Convert lists to tensors of shape (TotalScanlines, 768).
    small_sequences_tensor = torch.tensor(np.vstack(small_sequences), dtype=torch.float32)
    large_sequences_tensor = torch.tensor(np.vstack(large_sequences), dtype=torch.float32)

    return small_sequences_tensor, large_sequences_tensor
