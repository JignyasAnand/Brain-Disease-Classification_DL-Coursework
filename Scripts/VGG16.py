import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

base_dir = r'D:\KL\KL-3rd yr\Deep Learning\Data Set'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

img_height = 224
img_width = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

for layer in vgg16_base.layers:
    layer.trainable = False

model = Sequential([
    vgg16_base,
    TimeDistributed(Flatten()),  # Prepare for RNN
    LSTM(64),  # Use an LSTM layer or modify according to your needs
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')  # Assuming there are 5 classes
])
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # Modify this based on your requirements
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size
)
