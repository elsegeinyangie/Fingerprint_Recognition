import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



def train_model(model, X_train, y_train, X_test, y_test, model_name, 
                batch_size=32, epochs=20, model_save_dir="saved_models"):
    """Train the specified model for multiple outputs"""
    
    # Create model save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Define callbacks
    print("\nDefining callbacks for early stopping and model checkpointing...")
    callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    ),
    ModelCheckpoint(
        os.path.join(model_save_dir, f"{model_name}_best.keras"),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
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