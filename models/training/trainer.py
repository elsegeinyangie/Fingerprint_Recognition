import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def train_model(model, X_train, y_train, X_test, y_test, model_name, 
                batch_size=32, epochs=20, model_save_dir="saved_models"):
    """Train the specified model"""
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
        {"gender_output": y_train},  # Only using gender for now
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, {"gender_output": y_test}),
        callbacks=callbacks,
        verbose=1
    )
    return model, history