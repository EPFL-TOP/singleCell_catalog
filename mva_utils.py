import tensorflow as tf
import numpy as np
import os
import json


from tensorflow.keras import layers, models
from tensorflow.keras import regularizers


data_augmentation_simple = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),
])

data_augmentation_complex = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
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
def load_and_preprocess_images(json_dir, target_size, property):

    # Initialize lists to store images and labels
    images = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in filenames:
            keepfile=True
            if filename.endswith(".json"):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    data = json.load(f)
                    image_data = data['image_bf']
                    sub_lab = True
                    try:
                        for prop in property['conditions']:
                            if data[prop]!=property['conditions'][prop]:
                                keepfile=False
                            sub_lab = sub_lab*(data[prop]==property['conditions'][prop])
                            
                    except KeyError:
                        pass

                    if not keepfile: continue
                    label = data[property['main']]*sub_lab
                    processed_image = preprocess_image(image_data, target_size)
                    if len(processed_image) == 0: continue
                    # Append the image and label to lists
                    images.append(processed_image)
                    labels.append(1 if label == True else 0)  # Assuming 'dead_cell' is labeled as 1, 'live_cell' as 0
    return np.array(images), np.array(labels)



