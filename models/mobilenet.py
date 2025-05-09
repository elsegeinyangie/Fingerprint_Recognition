import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def mobilenetmodel(input_shape=(224, 224, 3)):
    # Base model - MobileNetV2
    
    print(f"Building MobileNetV2 model with input shape: {input_shape}")
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)

    # Output heads
    gender_output = Dense(2, activation='softmax', name='gender_output')(x)         # Male / Female
    hand_output = Dense(2, activation='softmax', name='hand_output')(x)             # Right / Left
    finger_output = Dense(5, activation='softmax', name='finger_output')(x)         # Thumb, Pointer, Middle, Ring, Pinky

    model = Model(inputs=inputs, outputs=[gender_output, hand_output, finger_output])
    return model
