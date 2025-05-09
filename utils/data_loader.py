import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import normalize_image, resize_image
import cv2



def extract_label(img_path, is_altered=False):
    # print(f"Extracting label from image path: {img_path}")
    filename, _ = os.path.splitext(os.path.basename(img_path))
    
    # split the filename into parts
    subject_id, attrs = filename.split('__')

    if is_altered:
        gender, hand, finger, _, _ = attrs.split('_')
    else:
        gender, hand, finger, _ = attrs.split('_')

    # encode labels
    gender = 0 if gender == 'M' else 1
    hand = 0 if hand == 'Left' else 1

    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}
    finger = finger_map.get(finger.lower(), -1)

    return gender, hand, finger




def loading_data(path, is_altered=False):
    # print(f"Loading data from path: {path}")
    data = []
    labels_gender, labels_hand, labels_finger = [], [], []

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.bmp'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # Skip files that failed to load

                gender, hand, finger = extract_label(file_path, is_altered)
                data.append(img)
                labels_gender.append(gender)
                labels_hand.append(hand)
                labels_finger.append(finger)

    return data, labels_gender, labels_hand, labels_finger


# def get_input_shape(model_name):
#     print(f"Getting input shape for model: {model_name}")
#     """Return the appropriate input shape for each model"""
#     # if model_name == 'lenet':
#     #     return (32, 32, 1)
#     # elif model_name == 'alexnet':
#     #     return (227, 227, 3)
#     if model_name in ['vgg', 'resnet', 'mobilenet']:
#         return (224, 224, 3)
#     # elif model_name =='googlenet':
#     #     return (299, 299, 3)
#     else:
#         raise ValueError(f"Unknown model type: {model_name}")

def get_input_shape(model_type):
    return (224, 224, 3)




def load_and_prepare_data(model_type, data_path="dataset"):

    # Define data paths
    real_path = os.path.join(data_path, "Real")
    altered_path = os.path.join(data_path, "Altered")

    # Load data
    real_data, real_gender, real_hand, real_finger = loading_data(real_path, is_altered=False)
    alt_data, alt_gender, alt_hand, alt_finger = loading_data(altered_path, is_altered=True)

    all_data = real_data + alt_data
    gender_labels = real_gender + alt_gender
    hand_labels = real_hand + alt_hand
    finger_labels = real_finger + alt_finger

    if len(all_data) == 0:
        raise ValueError("No data loaded.")

    # Shuffle data
    data = list(zip(all_data, gender_labels, hand_labels, finger_labels))
    np.random.shuffle(data)
    all_data, gender_labels, hand_labels, finger_labels = zip(*data)

    # Get the target shape based on model type
    target_shape = get_input_shape(model_type)

    # Pre-allocate array for processed images
    X = np.zeros((len(all_data), *target_shape), dtype=np.float32)

    # Process images
    for i, img in enumerate(all_data):
        if img is None or img.size == 0:
            print(f"Warning: Skipping empty image at index {i}")
            continue
        try:
            # Resize image first
            resized_img = resize_image(img, target_shape)
            
            # Now normalize the resized image
            normalized_img = normalize_image(resized_img, model_type)
            
            # Store in the pre-allocated array
            X[i] = normalized_img
        except Exception as e:
            print(f"Error processing image at index {i}: {e}")
            raise

    # Convert labels to NumPy arrays
    y_gender = np.array(gender_labels)
    y_hand = np.array(hand_labels)
    y_finger = np.array(finger_labels)

    # Train/test split
    return train_test_split(
        X, y_gender, y_hand, y_finger,
        test_size=0.2,
        random_state=42
    )

