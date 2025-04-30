import matplotlib.pyplot as plt



def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    results = model.evaluate(X_test, {"gender_output": y_test})
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
    return results

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['gender_output_accuracy'])
    plt.plot(history.history['val_gender_output_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['gender_output_loss'])
    plt.plot(history.history['val_gender_output_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()