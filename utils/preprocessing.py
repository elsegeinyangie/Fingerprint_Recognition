import cv2
import numpy as np

from tensorflow.keras.applications import (
    resnet50, vgg16, mobilenet, inception_v3
)

def normalize_image(img, model_type):
    """Normalize image based on model-specific preprocessing requirements."""
    print(f"Normalizing image for model type: {model_type}")
    # Add channel dimension if missing
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # Ensure float32 for safety
    img = img.astype(np.float32)

    if model_type == 'lenet' or model_type == 'alexnet':
        # Normalize to [0, 1]
        img /= 255.0

    elif model_type == 'resnet':
        # ResNet expects 3 channels and preprocess_input
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = resnet50.preprocess_input(img)

    elif model_type == 'vgg':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = vgg16.preprocess_input(img)

    elif model_type == 'mobilenet':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = mobilenet.preprocess_input(img)

    elif model_type == 'googlenet' or model_type == 'inception_v3':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = inception_v3.preprocess_input(img)

    else:
        raise ValueError(f"Unsupported model type in normalization: {model_type}")

    return img



def resize_image(image, target_shape):
    """Resize image and ensure it has the correct shape (H, W, C)"""
    print(f"Resizing image to target shape: {target_shape}")
    if image is None:
        raise ValueError("Cannot resize None image")
    
    # Extract target height and width
    target_height, target_width = target_shape[0], target_shape[1]
    
    # Resize the image
    img_resized = cv2.resize(image, (target_width, target_height))
    
    # Ensure image has the correct number of channels (1 for grayscale)
    if len(img_resized.shape) == 2:  # If image is 2D (height, width)
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    elif img_resized.shape[-1] != 1:  # If image has more than 1 channel
        # Take only the first channel for grayscale
        img_resized = img_resized[..., :1]
    
    return img_resized.astype(np.float32)