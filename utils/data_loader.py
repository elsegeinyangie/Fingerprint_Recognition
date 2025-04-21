import numpy as np
# import pandas as pd
# import seaborn as sns
# import tensorflow as tf
# import matplotlib.pyplot as plt
import os
import cv2

from preprocessing import resize_image_for_model, encode_attributes

############################################


# Function to extract labels for both real and altered fingerprint images
def extract_label(img_path, train=True):
    # Get the filename without the directory and extension
    filename, _ = os.path.splitext(os.path.basename(img_path))

    # Split the filename into subject ID and the rest (contains gender, hand side, finger, etc.)
    subject_id, etc = filename.split('__')

    # If the image is from the 'Altered' folder
    if train:
        # Split the remaining part into gender, left/right hand, finger type, and two extra fields
        gender, lr, finger, _, _ = etc.split('_')
    # If the image is from the 'Real' folder
    else:
        # Split into gender, left/right hand, finger type, and one extra field
        gender, lr, finger, _ = etc.split('_')

    gender_encoded, lr_encoded, finger_encoded = encode_attributes(gender, lr, finger)


    # Return only the gender as a NumPy array of type uint16
    return np.array([gender_encoded], dtype=np.uint16)


############################################


# Function to iterate through all the images in a folder, preprocess them, and load them with labels
def loading_data(path, train, model_type):
    print("Loading data from:", path)
    data = []  # Initialize an empty list to store [label, resized image] pairs

    for img in os.listdir(path):
        try:
            # Read the image in grayscale mode
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            # Resize the image according to the model type
            img_resized = resize_image_for_model(img_array, model_type)

            # Extract label (gender) for the image
            label = extract_label(os.path.join(path, img), train)

            # Append the label and resized image to the data list
            data.append([label[0], img_resized])

        except Exception as e:
            pass

    return data
