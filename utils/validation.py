import torch
import cv2
import numpy as np
import os
import time

def resize_cover(image, target_size, interpolation=cv2.INTER_NEAREST):
    """
    Resizes an image to target_size using a "cover" strategy (preserving aspect ratio,
    then center-cropping), similar to CSS's object-fit: cover.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    x = (new_w - target_w) // 2
    y = (new_h - target_h) // 2
    cropped = resized[y:y+target_h, x:x+target_w]
    return cropped

def preprocess_image(image_path):
    """
    Loads and preprocesses the test image for inference.
    Loads in color, resizes to 256x256 using cover strategy, normalizes,
    and flattens each row to obtain a tensor of shape (256, 768).
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_cover(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    img_norm = img.astype(np.float32) / 255.0
    flat_img = img_norm.reshape(256, -1)
    return torch.tensor(flat_img, dtype=torch.float32)

def decode_image_tensor(image_tensor, model, device):
    """
    Runs inference on a (256,768) tensor to produce a high-resolution output.
    Reshapes the model output from (256,768) back into (256,256,3).
    """
    src_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)
        output_sequence = model.decoder(encoder_outputs)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
        decoded_outputs = decoded_outputs / 20.0
        decoded_outputs = np.clip(decoded_outputs, 0, 1)
        decoded_image = decoded_outputs.reshape(256, 256, 3)
    return decoded_image

def save_decoded_image(decoded_image, output_path):
    """
    Saves the decoded high-resolution RGB image.
    Converts the normalized [0,1] image to uint8 and saves as a PNG.
    """
    image_to_save = (decoded_image * 255.0).astype(np.uint8)
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
    image_tensor = preprocess_image(test_image_path)
    decoded_image = decode_image_tensor(image_tensor, model, device)
    save_decoded_image(decoded_image, output_path)

def preprocess_patch(patch):
    """
    Preprocesses a patch for inference.
    Upscales the patch to 256x256 using cover strategy, normalizes to [0,1],
    and flattens to a tensor of shape (256,768).
    """
    patch_resized = resize_cover(patch, (256, 256), interpolation=cv2.INTER_NEAREST)
    patch_norm = patch_resized.astype(np.float32) / 255.0
    patch_flat = patch_norm.reshape(256, -1)
    return torch.tensor(patch_flat, dtype=torch.float32)

def validate_model_patches(model, device):
    """
    Processes the test image in patches.
    Each patch is upscaled to 256x256 using cover strategy, processed through the model,
    and the final image is reassembled and saved with a unique filename.
    """
    test_image_path = "dataset/test_set/test.png"
    output_dir = "dataset/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"test_output_{int(time.time())}.png")
    
    img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {test_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    H, W, _ = img.shape
    patch_size = 128
    n_rows = H // patch_size
    n_cols = W // patch_size

    patches = []
    for row in range(n_rows):
        row_patches = []
        for col in range(n_cols):
            patch = img[row * patch_size:(row + 1) * patch_size,
                        col * patch_size:(col + 1) * patch_size, :]
            row_patches.append(patch)
        patches.append(row_patches)
    
    decoded_patches = []
    for row_patches in patches:
        decoded_row = []
        for patch in row_patches:
            patch_tensor = preprocess_patch(patch)
            decoded_patch = decode_image_tensor(patch_tensor, model, device)
            decoded_row.append(decoded_patch)
        decoded_patches.append(decoded_row)
    
    final_image_rows = [np.hstack(decoded_row) for decoded_row in decoded_patches]
    final_image = np.vstack(final_image_rows)
    save_decoded_image(final_image, output_path)
