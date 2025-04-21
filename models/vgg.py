from tensorflow.python.keras import layers, models

def vgg(input_shape):
    """
    Builds a simplified VGG-like model for multi-output classification.
    """
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Multi-output heads
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(model.output)
    hand_output = layers.Dense(2, activation='softmax', name='hand_output')(model.output)
    finger_output = layers.Dense(5, activation='softmax', name='finger_output')(model.output)

    final_model = models.Model(inputs=model.input, outputs=[gender_output, hand_output, finger_output])

    final_model.compile(optimizer='adam',
                        loss={'gender_output': 'sparse_categorical_crossentropy',
                            'hand_output': 'sparse_categorical_crossentropy',
                            'finger_output': 'sparse_categorical_crossentropy'},
                        metrics=['accuracy'])

    return final_model
