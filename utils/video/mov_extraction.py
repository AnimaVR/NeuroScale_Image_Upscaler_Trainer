import os
import cv2

def find_files(folder_path):
    mov_path, mp4_path = None, None

    # Define the directories for frames
    small_frames_dir = os.path.join(folder_path, 'frames_small')
    large_frames_dir = os.path.join(folder_path, 'frames_large')

    # Create the directories if they don't exist
    os.makedirs(small_frames_dir, exist_ok=True)
    os.makedirs(large_frames_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        if file.endswith('.mov'):
            mov_path = os.path.join(folder_path, file)
        elif file.endswith('.mp4'):
            mp4_path = os.path.join(folder_path, file)

    return mov_path, mp4_path, small_frames_dir, large_frames_dir


def extract_frames(video_path, output_dir_small, output_dir_large, small_size=(64, 64), large_size=(256, 256)):
    """
    Extracts frames from a video file, resizes them to two resolutions, and saves them.
    """
    # Ensure the output directories exist (redundant if find_files has already created them)
    os.makedirs(output_dir_small, exist_ok=True)
    os.makedirs(output_dir_large, exist_ok=True)

    # Check if frames already exist and match in count
    existing_small = sorted(os.listdir(output_dir_small))
    existing_large = sorted(os.listdir(output_dir_large))

    if existing_small and existing_large and len(existing_small) == len(existing_large):
        print(f"Frames already extracted for {video_path}. Using existing frames.")
        return  # Skip re-extraction

    print(f"Extracting frames from {video_path}...")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # No more frames

        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frames
        small_frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_CUBIC)
        large_frame = cv2.resize(frame, large_size, interpolation=cv2.INTER_CUBIC)

        # Save frames (convert back from RGB to BGR for saving)
        small_frame_path = os.path.join(output_dir_small, f"frame_{frame_count:04d}.png")
        large_frame_path = os.path.join(output_dir_large, f"frame_{frame_count:04d}.png")

        cv2.imwrite(small_frame_path, cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(large_frame_path, cv2.cvtColor(large_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")
