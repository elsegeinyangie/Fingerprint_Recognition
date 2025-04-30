# utils/data_loader.py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils.preprocessing import normalize_image
import cv2

def loading_data(path, train=True, model_type=None):
    """
    Load data from directory, including subdirectories.
    """
    data = []
    label = 0 if os.path.basename(path).lower() == "real" else 1
    
    # Print debug info
    print(f"\nLoading data from: {path}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not os.path.exists(path):
        print(f"ERROR: Path does not exist: {path}")
        return data

    for root, dirs, files in os.walk(path):
        print(f"Scanning directory: {root}")
        print(f"Found {len(files)} files")
        
        for file in files:
            if file.lower().endswith('.bmp'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Failed to load image: {file_path}")
                        continue
                        
                    data.append((label, img))
                    print(f"Successfully loaded: {file_path}")
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Total loaded from {path}: {len(data)} images")
    return data
    
    # Process the directory and all its subdirectories
    data = process_directory(path)
    
    print(f"Loaded {len(data)} images from {path} and its subdirectories")
    return data






def get_input_shape(model_name):
    """Return the appropriate input shape for each model"""
    if model_name == 'lenet':
        return (32, 32, 1)
    elif model_name == 'alexnet':
        return (227, 227, 1)
    elif model_name in ['vgg', 'googlenet', 'resnet']:
        return (224, 224, 1)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def load_and_prepare_data(model_type, data_path="dataset"):
    """Load and prepare the dataset for training"""
    print("\n" + "="*50)
    print("Starting data loading process")
    print("="*50)
    
    # Load real fingerprints
    real_path = os.path.join(data_path, "Real")
    print(f"\nLoading REAL fingerprints from: {real_path}")
    real_data = loading_data(real_path, train=True, model_type=model_type)

    # Load altered fingerprints
    altered_path = os.path.join(data_path, "Altered")
    print(f"\nLoading ALTERED fingerprints from: {altered_path}")
    altered_data = loading_data(altered_path, train=True, model_type=model_type)
    
    # Combine data
    all_data = real_data + altered_data
    print(f"\nTotal images loaded: {len(all_data)}")
    print(f"Real: {len(real_data)}, Altered: {len(altered_data)}")
    
    if len(all_data) == 0:
        raise ValueError("No images loaded! Check your dataset paths and file extensions.")
    
    # Check if we have any data
    if len(all_data) == 0:
        raise ValueError(f"No data found in {data_path}/real or {data_path}/altered directories. Please check your dataset paths.")
    
    # Print data statistics
    print(f"Loaded {len(real_data)} real images and {len(altered_data)} altered images.")
    
    # Shuffle the data
    np.random.shuffle(all_data)
    
    # Separate features and labels
    X = np.array([normalize_image(i[1]) for i in all_data])
    y = np.array([i[0] for i in all_data])
    
    # Reshape X to include channel dimension and resize if needed
    target_shape = get_input_shape(model_type)
    if X.shape[1:] != target_shape[:2]:  # If images aren't already the right size
        X_resized = np.zeros((X.shape[0], *target_shape))
        for i in range(X.shape[0]):
            img = X[i]
            # Resize while maintaining single channel
            img_resized = cv2.resize(img, target_shape[:2])
            if len(img_resized.shape) == 2:
                img_resized = np.expand_dims(img_resized, axis=-1)
            X_resized[i] = img_resized
        X = X_resized
    else:
        X = X.reshape(X.shape + (1,))  # Just add channel dimension
    
    # Convert labels to categorical (one-hot encoding)
    y = to_categorical(y, num_classes=2)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test