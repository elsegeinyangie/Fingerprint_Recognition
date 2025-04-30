from keras import layers, models

# Builds a modified AlexNet model for multi-output classification.
def alexnet(input_shape):

    model = models.Sequential()

    # 1st Convolutional Layer
    model.add(layers.Conv2D(96, (11, 11), strides=(4,4), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # 2nd Convolutional Layer
    model.add(layers.Conv2D(256, (5, 5), padding="same", activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # 3rd, 4th, and 5th Convolutional Layers
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    model.add(layers.Flatten())

    # Fully Connected Layers
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
