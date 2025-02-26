import torch
import cv2
import numpy as np
import os
import time

# dont forget to reduce the frame size and inference @ frame size with linear blend between frames to complete image like we do face shapes


def preprocess_image(image_path):
    """
    Loads and preprocesses the test image for inference.
    Loads in color, resizes to 256x256, normalizes, and flattens each row.
    Returns a tensor of shape (256, 768).
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 256x256 using nearest-neighbor to preserve pixelation
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    # Normalize pixel values to [0,1]
    img_norm = img.astype(np.float32) / 255.0
    # Flatten each row: (256, 256, 3) -> (256, 768)
    flat_img = img_norm.reshape(256, -1)
    return torch.tensor(flat_img, dtype=torch.float32)

def decode_image_tensor(image_tensor, model, device):
    """
    Runs inference on a (256, 768) tensor to produce a high-resolution output.
    Reshapes the model output from (256, 768) back into (256, 256, 3).
    """
    # Ensure tensor is on the correct device and add batch dimension
    src_tensor = image_tensor.unsqueeze(0).to(device)  # shape: (1, 256, 768)
    with torch.no_grad():
        # Pass through model
        encoder_outputs = model.encoder(src_tensor)
        output_sequence = model.decoder(encoder_outputs)
        # Remove batch dimension and move to CPU; expected shape: (256, 768)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
        decoded_outputs = decoded_outputs / 10.0
        # Clip values to ensure they're in valid range [0,1]
        decoded_outputs = np.clip(decoded_outputs, 0, 1)
        # Reshape to (256, 256, 3)
        decoded_image = decoded_outputs.reshape(256, 256, 3)
    return decoded_image

def save_decoded_image(decoded_image, output_path):
    """
    Saves the decoded high-resolution RGB image.
    Converts the normalized [0,1] image to uint8 and saves as a PNG.
    """
    # Convert from float [0,1] to uint8 [0,255]
    image_to_save = (decoded_image * 255.0).astype(np.uint8)
    # Convert from RGB to BGR for OpenCV saving
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_to_save)
    print(f"Saved upscaled image to {output_path}")

def validate_model(model, device):
    """
    Runs inference on the test image (assumed to be at dataset/test_set/test.png) and saves the result.
    """
    test_image_path = "dataset/test_set/test.png"
    output_dir = "dataset/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_output.png")
    # Preprocess test image
    image_tensor = preprocess_image(test_image_path)
    # Run inference
    decoded_image = decode_image_tensor(image_tensor, model, device)
    # Save the upscaled output
    save_decoded_image(decoded_image, output_path)



def validate_model_patches(model, device):
    """
    Runs inference on the test image by processing it in 64x64 patches.
    Each patch is upscaled to 256x256, processed through the model, and then reassembled
    into the final high-resolution output image. The final image is saved with a unique name.
    """
    test_image_path = "dataset/test_set/test.png"
    output_dir = "dataset/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    # Create a unique filename using a timestamp
    output_path = os.path.join(output_dir, f"test_output_{int(time.time())}.png")
    
    # Load the full test image in color and convert to RGB
    img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {test_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Determine image dimensions and patch extraction parameters
    H, W, _ = img.shape
    patch_size = 128
    n_rows = H // patch_size
    n_cols = W // patch_size

    # Extract non-overlapping 64x64 patches
    patches = []
    for row in range(n_rows):
        row_patches = []
        for col in range(n_cols):
            patch = img[row * patch_size:(row + 1) * patch_size,
                        col * patch_size:(col + 1) * patch_size, :]
            row_patches.append(patch)
        patches.append(row_patches)
    
    # Process each patch: upscale, run inference, and collect decoded patches
    decoded_patches = []
    for row_patches in patches:
        decoded_row = []
        for patch in row_patches:
            patch_tensor = preprocess_patch(patch)  # (256,768)
            decoded_patch = decode_image_tensor(patch_tensor, model, device)  # (256,256,3)
            decoded_row.append(decoded_patch)
        decoded_patches.append(decoded_row)
    
    # Reassemble the final image by stitching the decoded patches together
    # Each decoded patch is (256,256,3)
    final_image_rows = [np.hstack(decoded_row) for decoded_row in decoded_patches]
    final_image = np.vstack(final_image_rows)
    
    # Save the final upscaled image
    save_decoded_image(final_image, output_path)


def preprocess_patch(patch):
    """
    Preprocesses a 64x64 patch for inference.
    Upscales the patch to 256x256 using nearest-neighbor interpolation,
    normalizes the pixel values to [0,1], and flattens each row to obtain a tensor of shape (256, 768).
    """
    # Upscale patch from 64x64 to 256x256
    patch_resized = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_NEAREST)
    # Normalize pixel values to [0,1]
    patch_norm = patch_resized.astype(np.float32) / 255.0
    # Flatten each row: from (256, 256, 3) to (256, 768)
    patch_flat = patch_norm.reshape(256, -1)
    return torch.tensor(patch_flat, dtype=torch.float32)