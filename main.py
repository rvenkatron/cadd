from google.colab import drive
drive.mount('/content/drive')

train_dir = '/content/drive/My Drive/Datasets/TrainingDataset'
test_dir = '/content/drive/My Drive/Datasets/TestingDataset'

import os
print("Train Directory Contents:", os.listdir(train_dir))
print("Test Directory Contents:", os.listdir(test_dir))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),  # Updated to 299x299
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))  # Updated to 299x299

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001)

history = model.fit(train_generator, epochs=10, callbacks=[reduce_lr])

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

import numpy as np
from tensorflow.keras.preprocessing import image

