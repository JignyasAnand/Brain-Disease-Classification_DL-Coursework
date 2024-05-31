import tensorflow as tf
import os
from preprocess import structure_datasets,get_ds_splits

# Sequential model to binary classification
'''
def ann(train_generator,test_generator):
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')  # Assuming 1 output class
    ])

    model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model1.fit(
        train_generator,
        epochs=5,  # You can adjust the number of epochs as needed
        validation_data=test_generator,
        batch_size=32
    )
'''
# Sequential model to classify project  data by adding various optimization techniques like ADAM, SGD, RMSPROP
'''
def ann(train_generator,test_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming 5 output classes
    ])
    
    print("Adam")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(
        train_generator,
        epochs=1,  # You can adjust the number of epochs as needed
        validation_data=test_generator,
        batch_size=32
    )
    
    print("SGD")
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history = model.fit(
        train_generator,
        epochs=1,  # You can adjust the number of epochs as needed
        validation_data=test_generator,
        batch_size=32
    )
    
    print("RMSProp")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    history = model.fit(
        train_generator,
        epochs=1,  # You can adjust the number of epochs as needed
        validation_data=test_generator,
        batch_size=32
    )

#    # SGD
#     print("SGD")
#     model2 = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming 5 output classes
#     ])
#
#     model2.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
#
#     history = model2.fit(
#         train_generator,
#         epochs=5,  # You can adjust the number of epochs as needed
#         validation_data=test_generator,
#         batch_size=32
#     )
#
#     #rmsprop
#     print("rmsprop")
#     model3 = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming 5 output classes
#     ])
#
#     model3.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
#
#     history = model3.fit(
#         train_generator,
#         epochs=5,  # You can adjust the number of epochs as needed
#         validation_data=test_generator,
#         batch_size=32
#     )'''

# Sequential model to classify project data in to multiple classes

def ann(train_generator,test_generator):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 output classes
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        train_generator,
        epochs=1,  # You can adjust the number of epochs as needed
        validation_data=test_generator,
        batch_size=32
    )