import os
import json
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


# Define the paths
json_dir = '/mnt/sdc1/data/singleCell_training/'
if not os.path.exists('/mnt/sdc1/data/singleCell_training'):
    json_dir = '/data/singleCell_training/'


# Target image size (height, width)
target_size = (150, 150)


#__________________________________________________________________________________
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


#__________________________________________________________________________________
def preprocess_image(image, target_size):
    # Calculate padding
    image = np.clip(image, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    image = np.array(image, dtype=np.int16)

    if image.shape[0]>target_size[0] or image.shape[1]>target_size[1]: return []

    delta_w = target_size[1] - image.shape[1]
    delta_h = target_size[0] - image.shape[0]
    pad_width = delta_w // 2
    pad_height = delta_h // 2

    padding = ((pad_height, pad_height), (pad_width, pad_width))

    # Check if the padding difference is odd and distribute padding accordingly
    if delta_w % 2 != 0:
        padding = ((pad_height, pad_height), (pad_width, pad_width+1))

    if delta_h % 2 != 0:
        padding = ((pad_height, pad_height+1), (pad_width, pad_width))

    if delta_h % 2 != 0 and delta_w % 2 != 0:
        padding = ((pad_height, pad_height+1), (pad_width, pad_width+1))
    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)

    padded_image = padded_image / np.max(padded_image)
    #padded_image = np.expand_dims(padded_image, axis=-1)

    return padded_image

#__________________________________________________________________________________
def load_and_preprocess_images(json_dir, target_size=(150, 150)):

    # Initialize lists to store images and labels
    images = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    data = json.load(f)
                    image_data = data['image_bf']
                    label = data['alive']
                    
                    processed_image = preprocess_image(image_data, target_size)
                    if len(processed_image) == 0: continue
                    # Append the image and label to lists
                    images.append(processed_image)
                    labels.append(1 if label == True else 0)  # Assuming 'dead_cell' is labeled as 1, 'live_cell' as 0
    return np.array(images), np.array(labels)



images, labels = load_and_preprocess_images(json_dir)
images = np.expand_dims(images, axis=-1)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])


model = models.Sequential([
    data_augmentation,

    layers.Input(shape=(150, 150, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model = models.Sequential([
    data_augmentation,
    #layers.Input(shape=(150, 150, 1)),

    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    lr_scheduler,
]

# Train the model
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks)




model.save('cell_classifier_model.keras')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('training_acc_loss.png')