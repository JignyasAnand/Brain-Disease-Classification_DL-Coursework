# Binary classification with mini-batch evaluations
'''
import tensorflow as tf
def cnn(train_generator,test_generator):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(1, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        batch_size=32
    )
'''
# CNN with regularization

import tensorflow as tf
from tensorflow.python.keras import regularizers


def cnn(train_generator,test_generator):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3),kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="tanh"),
        tf.keras.layers.Dense(1, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        epochs=5,
        validation_data=test_generator,
        batch_size=32
    )

# import tensorflow as tf
# import os
# from InClass_3 import generate_train_test_images
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#
# base_dir = r'C:\Users\SAI SURYA TEJA\PycharmProjects\Deep Learning\InClass\Images'
# train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')
# valid_dir = os.path.join(base_dir, 'valid')
#
# train_generator, test_generator, valid_generator = generate_train_test_images(
#     train_dir, test_dir, valid_dir, batch_size=32, img_height=124, img_width=124
# )
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Reshape((124, 124*3), input_shape=(124, 124, 3)),
#     tf.keras.layers.LSTM(64, return_sequences=True)
#
#     tf.keras.layers.LSTM(32),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(5, activation='softmax')
# ])
#
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validati`1on_data=valid_generator
# )