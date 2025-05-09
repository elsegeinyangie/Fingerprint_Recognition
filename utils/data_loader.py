import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.preprocessing import normalize_image, resize_image
import cv2
from tensorflow.keras.utils import to_categorical


def extract_label(img_path, is_altered=False):
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


def load_and_prepare_data(model_type, data_path="dataset", batch_size=16):
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

    # Get the input shape
    target_shape = (224, 224, 3)

    # Create a tf.data.Dataset from the numpy arrays
    X = np.zeros((len(all_data), *target_shape), dtype=np.float32)
    for i, img in enumerate(all_data):
        if img is None or img.size == 0:
            print(f"[WARNING] Skipping empty image at index {i}")
            continue
        try:
            resized_img = resize_image(img, target_shape)
            normalized_img = normalize_image(resized_img, model_type)
            X[i] = normalized_img
        except Exception as e:
            print(f"[ERROR] Processing image at index {i}: {e}")
            continue

    print(f"[INFO] Finished processing images. Total processed: {len(X)}")
    
    # Convert labels to numpy arrays
    y_gender = np.array(gender_labels)
    y_hand = np.array(hand_labels)
    y_finger = np.array(finger_labels)
    
    # One-hot encode labels
    print("[INFO] One-hot encoding labels...")
    y_gender = to_categorical(y_gender)
    y_hand = to_categorical(y_hand)
    y_finger = to_categorical(y_finger)
    
    print("[INFO] Splitting data into training and testing sets...")
    X_train, X_test, y_gender_train, y_gender_test, y_hand_train, y_hand_test, y_finger_train, y_finger_test = train_test_split(
        X, y_gender, y_hand, y_finger,
        test_size=0.2,
        random_state=42
    )

    print("[INFO] Data loading and preparation complete.")

    # Create a tf.data.Dataset from the training and test data
    def create_dataset(X_data, y_gender_data, y_hand_data, y_finger_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((
            X_data, {
                'gender_output': y_gender_data,
                'hand_output': y_hand_data,
                'finger_output': y_finger_data
            }
        ))
        dataset = dataset.shuffle(buffer_size=1000)  # Shuffle the data
        dataset = dataset.batch(batch_size)  # Set the batch size
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Prefetch for better performance
        return dataset

    # Return the dataset instead of numpy arrays
    train_dataset = create_dataset(X_train, y_gender_train, y_hand_train, y_finger_train, batch_size)
    test_dataset = create_dataset(X_test, y_gender_test, y_hand_test, y_finger_test, batch_size)
    
    return train_dataset, test_dataset
