import cv2
import numpy as np

from tensorflow.keras.applications import (
    resnet50, vgg16, mobilenet
)

def normalize_image(img, model_type):
    """Normalize image based on model-specific preprocessing requirements."""

    # Add channel dimension if missing (for grayscale images)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # Ensure the image is of type float32
    img = img.astype(np.float32)

    if model_type == 'resnet':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = resnet50.preprocess_input(img)

    elif model_type == 'vgg':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = vgg16.preprocess_input(img)

    elif model_type == 'mobilenet':
        img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
        img = mobilenet.preprocess_input(img)

    else:
        raise ValueError(f"Unsupported model type for normalization: {model_type}")

    return img



def resize_image(image, target_shape):
    #Resize image to the target shape (H, W, C).
    height, width = target_shape[0], target_shape[1]
    image_resized = cv2.resize(image, (width, height))

    # Ensure the resized image has 3 channels
    if image_resized.ndim == 2:
        image_resized = np.expand_dims(image_resized, axis=-1)

    return image_resized