import cv2
import numpy as np

def normalize_image(image):
    """
    Normalize an image to have values between 0 and 1.
    """
    # Check if image is already normalized
    if image.max() > 1.0:
        return image / 255.0
    return image

def resize_image(image, target_shape):
    """Resize an image to the target shape"""
    img_resized = cv2.resize(image, (target_shape[1], target_shape[0]))
    
    # Ensure we have a channel dimension
    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
        
    return img_resized