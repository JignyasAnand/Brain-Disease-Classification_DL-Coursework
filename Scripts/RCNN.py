import tensorflow as tf
import os
from Augmentation import generate_train_test_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

base_dir = r'D:\KL\KL-3rd yr\Deep Learning\Data Set'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

train_generator, test_generator, valid_generator = generate_train_test_images(
    train_dir, test_dir, valid_dir, batch_size=32, img_height=124, img_width=124
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),


    tf.keras.layers.Reshape((13, 128 * 13)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),

    # Fully connected layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)
