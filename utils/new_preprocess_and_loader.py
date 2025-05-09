import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
# from utils.preprocessing import normalize_image, resize_image
from utils.data_loader import loading_data, get_input_shape
import matplotlib.pyplot as plt
from models.resnet import resnet
from models.vgg import vgg
from models.mobilenet import mobilenetmodel
from tensorflow.keras.applications import resnet50, vgg16, mobilenet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical




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

    # TEMPORARY: limit data to avoid kernel crash
    MAX_SAMPLES = 100  # You can increase this if things work
    all_data = all_data[:MAX_SAMPLES]
    gender_labels = gender_labels[:MAX_SAMPLES]
    hand_labels = hand_labels[:MAX_SAMPLES]
    finger_labels = finger_labels[:MAX_SAMPLES]

    # Shuffle data
    data = list(zip(all_data, gender_labels, hand_labels, finger_labels))
    np.random.shuffle(data)
    all_data, gender_labels, hand_labels, finger_labels = zip(*data)

    # Get the input shape
    target_shape = (224, 224, 3)

    X = np.zeros((len(all_data), *target_shape), dtype=np.float32)

    for i, img in enumerate(all_data):
        if img is None or img.size == 0:
            print(f"[WARNING] Skipping empty image at index {i}")
            continue
        try:
            resized_img = resize_image(img, target_shape)
            normalized_img = normalize_image(resized_img, model_type)
            X[i] = normalized_img
            if i % 20 == 0:
                print(f"[INFO] Processed {i}/{len(all_data)} images")
        except Exception as e:
            print(f"[ERROR] Processing image at index {i}: {e}")
            continue
    
    print(f"[INFO] Finished processing images. Total processed: {len(X)}")
    print("INFO] Converting data to numpy arrays...")
    # Convert labels to numpy arrays
    y_gender = np.array(gender_labels)
    y_hand = np.array(hand_labels)
    y_finger = np.array(finger_labels)
    
    # One-hot encode labels
    print("[INFO] One-hot encoding labels...")
    y_gender = to_categorical(y_gender)
    y_hand = to_categorical(y_hand)
    y_finger = to_categorical(y_finger)

    return train_test_split(X, y_gender, y_hand, y_finger, test_size=0.2, random_state=42)




def resize_image(image, target_shape):
    """
    Resize image to the target shape (H, W, C).
    """
    height, width = target_shape[0], target_shape[1]
    image_resized = cv2.resize(image, (width, height))
    
    # Ensure the resized image has 3 channels
    if image_resized.ndim == 2:
        image_resized = np.expand_dims(image_resized, axis=-1)

    return image_resized


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


def get_model(model_name, input_shape):
    """Get the specified model (ResNet, VGG, MobileNet)"""
    if model_name == 'resnet':
        return resnet(input_shape)
    elif model_name == 'vgg':
        return vgg(input_shape)
    elif model_name == 'mobilenet':
        return mobilenetmodel(input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")



def train_model(model, X_train, y_train, X_test, y_test, model_name, 
                batch_size=32, epochs=20, model_save_dir="saved_models"):
    """Train the specified model for multiple outputs"""
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_gender_output_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(model_save_dir, f"{model_name}_best.h5"),
            monitor='val_gender_output_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Train model
    history = model.fit(
        X_train,
        {
            "gender_output": y_train[0],
            "hand_output": y_train[1],
            "finger_output": y_train[2]
        },
        validation_data=(X_test, {
            "gender_output": y_test[0],
            "hand_output": y_test[1],
            "finger_output": y_test[2]
        }),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on gender, hand, and finger predictions"""
    results = model.evaluate(X_test, {
        "gender_output": y_test[0], 
        "hand_output": y_test[1],
        "finger_output": y_test[2]
    })
    
    print(f"Test Loss (Gender): {results[0]}")
    print(f"Test Accuracy (Gender): {results[1]}")
    print(f"Test Loss (Hand): {results[2]}")
    print(f"Test Accuracy (Hand): {results[3]}")
    print(f"Test Loss (Finger): {results[4]}")
    print(f"Test Accuracy (Finger): {results[5]}")
    
    return results

def plot_history(history):
    """Plot training history for accuracy and loss of all outputs"""
    plt.figure(figsize=(15, 5))

    # Plot accuracy for all three outputs
    plt.subplot(1, 3, 1)
    plt.plot(history.history['gender_output_accuracy'])
    plt.plot(history.history['val_gender_output_accuracy'])
    plt.title('Gender Output Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['hand_output_accuracy'])
    plt.plot(history.history['val_hand_output_accuracy'])
    plt.title('Hand Output Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['finger_output_accuracy'])
    plt.plot(history.history['val_finger_output_accuracy'])
    plt.title('Finger Output Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss for all three outputs
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['gender_output_loss'])
    plt.plot(history.history['val_gender_output_loss'])
    plt.title('Gender Output Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 2)
    plt.plot(history.history['hand_output_loss'])
    plt.plot(history.history['val_hand_output_loss'])
    plt.title('Hand Output Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(history.history['finger_output_loss'])
    plt.plot(history.history['val_finger_output_loss'])
    plt.title('Finger Output Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


