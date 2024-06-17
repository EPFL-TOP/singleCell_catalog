import tensorflow as tf
import numpy as np
import os
import json


from tensorflow.keras import layers, models
from tensorflow.keras import regularizers


data_augmentation_simple = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

data_augmentation_complex = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])


model_simple = models.Sequential([
    data_augmentation_simple,

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



model_complex = models.Sequential([
    data_augmentation_complex,
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


base_model_tl = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

model_tl = models.Sequential([
    data_augmentation_simple,
    layers.Input(shape=(150, 150, 3)),  # Convert grayscale to RGB if needed

    base_model_tl,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

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



