import tensorflow as tf
import os
from CNN_2 import generate_train_test_images

# Adjust the TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Define directories
base_dir = r'D:\KL\KL-3rd yr\Deep Learning\Data Set'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# Adjust these parameters to fit the VGG16 input size
img_height = 224
img_width = 224

# Generate the image datasets
train_generator, test_generator, valid_generator = generate_train_test_images(
    train_dir, test_dir, valid_dir, batch_size=32, img_height=img_height, img_width=img_width
)

# Load the VGG16 model, excluding its top layer (the classification layers)
base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

# Freeze the layers of the base_model
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="tanh"),
    tf.keras.layers.Dense(7, activation="softmax")  # Assuming you have 5 classes
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=valid_generator,
    batch_size=32
)
