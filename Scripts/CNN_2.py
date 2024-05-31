import tensorflow as tf
from Augmentation import *
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
        epochs=5,
        validation_data=test_generator,
        batch_size=32
    )

if __name__ == '__main__':
    train_dir=''
    test_dir=''
    valid_dir=''
    batch_size=32
    train_gen,test_gen,valid_gen=generate_train_test_images(train_dir,test_dir,valid_dir,batch_size)
    cnn(train_gen,test_gen)

