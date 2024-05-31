import tensorflow as tf
def vgNet(train_generator,test_generator):

    model = tf.keras.models.Sequential([
        tf.keras.applications.vgg19.VGG19(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3),
        ),
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
