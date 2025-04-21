from keras import layers, models
from keras.applications import InceptionV3

def googlenet(input_shape):
    """
    Builds an InceptionV3 (GoogleNet) model for multi-output classification.
    """
    base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape, pooling='avg')

    x = base_model.output

    # Multi-output heads
    gender_output = layers.Dense(2, activation='softmax', name='gender_output')(x)
    hand_output = layers.Dense(2, activation='softmax', name='hand_output')(x)
    finger_output = layers.Dense(5, activation='softmax', name='finger_output')(x)

    final_model = models.Model(inputs=base_model.input, outputs=[gender_output, hand_output, finger_output])

    final_model.compile(optimizer='adam',
                        loss={'gender_output': 'sparse_categorical_crossentropy',
                            'hand_output': 'sparse_categorical_crossentropy',
                            'finger_output': 'sparse_categorical_crossentropy'},
                        metrics=['accuracy'])

    return final_model
