import matplotlib.pyplot as plt



def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Evaluate for all three outputs: gender, hand, and finger
    print(f"Evaluating model{model}...")
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
    """Plot training history for all three outputs"""
    print("Plotting training history...")
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
