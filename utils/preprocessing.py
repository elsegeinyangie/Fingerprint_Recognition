# imports
from PIL import Image
import numpy as np

############################################

# Resize the image according to the input size required by the model.
def resize_image_for_model(image, model_type):

    if model_type.lower() == 'lenet':
        target_size = (32, 32)
    elif model_type.lower() == 'alexnet':
        target_size = (227, 227)
    elif model_type.lower() in ['vgg', 'googlenet', 'resnet']:
        target_size = (224, 224)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Supported: lenet, alexnet, vgg, googlenet, resnet.")

    # Convert NumPy array to PIL Image for resizing
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize(target_size)

    # Convert back to NumPy array (if you want to stay consistent with OpenCV images)
    return np.array(img_resized)

############################################


# Helper function to encode gender, hand side, and finger type
def encode_attributes(gender, lr, finger):
    # Convert gender to numerical value
    gender = 0 if gender == 'M' else 1

    # Convert hand side (Left/Right) to numerical
    lr = 0 if lr == 'Left' else 1

    # Map the finger type to a numerical value
    finger_mapping = {
        'thumb': 0,
        'index': 1,
        'middle': 2,
        'ring': 3,
        'little': 4
    }
    finger = finger_mapping.get(finger, -1)  # -1 if finger type is unknown

    return gender, lr, finger


def normalize_image(img):
    return img / 255.0
