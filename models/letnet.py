from tensorflow.python.keras import layers, models

def lenet(input_shape):
    """
    Builds a modified LeNet-5 model for multi-output classification:
    gender, left/right hand, and finger type.
    """
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D())

    # Second convolutional layer
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))

    # Now create three separate outputs
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(model.output)
    hand_output = layers.Dense(2, activation='softmax', name='hand_output')(model.output)
    finger_output = layers.Dense(5, activation='softmax', name='finger_output')(model.output)

    # Combine into a single model
    final_model = models.Model(inputs=model.input, outputs=[gender_output, hand_output, finger_output])

    # Compile the model
    final_model.compile(optimizer='adam',
                        loss={'gender_output': 'sparse_categorical_crossentropy',
                            'hand_output': 'sparse_categorical_crossentropy',
                            'finger_output': 'sparse_categorical_crossentropy'},
                        metrics=['accuracy'])

    return final_model
