import tensorflow as tf
import os
from Augmentation import generate_train_test_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

base_dir = r'C:\Users\SAI SURYA TEJA\PycharmProjects\Deep Learning\InClass\Images'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

train_generator, test_generator, valid_generator = generate_train_test_images(
    train_dir, test_dir, valid_dir, batch_size=32, img_height=124, img_width=124
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((124, 124*3), input_shape=(124, 124, 3)),
    tf.keras.layers.SimpleRNN(64, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)
