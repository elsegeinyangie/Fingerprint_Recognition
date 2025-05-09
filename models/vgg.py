import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input

def vgg(input_shape=(224, 224, 3)):
    
    print(f"Building VGG16 model with input shape: {input_shape}")
    
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # Optional: freeze VGG16 layers

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output heads
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(x)
    hand_output = layers.Dense(2, activation='softmax', name='hand_output')(x)
    finger_output = layers.Dense(5, activation='softmax', name='finger_output')(x)

    model = Model(inputs=inputs, outputs=[gender_output, hand_output, finger_output])
    return model