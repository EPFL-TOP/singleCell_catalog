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

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Stop training when a monitored metric has stopped improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Define the paths
json_dir = '/mnt/sdc1/data/singleCell_training/'
if not os.path.exists('/mnt/sdc1/data/singleCell_training'):
    json_dir = '/data/singleCell_training/'




# Target image size (height, width)
target_size = (150, 150)

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

# Function to load JSON data and convert to numpy arrays
def load_and_preprocess_images(json_dir, target_size=(150, 150)):

    # Initialize lists to store images and labels
    sequences = []
    labels = []

    # /data/singleCell_training/ppf003/ppf003_well1/ppf003_xy001
    for expname in os.listdir(json_dir):
        for well_name in os.listdir(os.path.join(json_dir,expname)):
            for pos_name in os.listdir(os.path.join(json_dir, expname, well_name)):
                sequence = []
                label    = []
                frame    = []
                for filename in sorted(os.listdir(os.path.join(json_dir, expname, well_name, pos_name))):
                    if filename.endswith('.json'):
                        with open(os.path.join(json_dir, expname, well_name, pos_name, filename), 'r') as f:
                            data = json.load(f)
                            image_data = data['image_bf']
                            alive = data['alive']
                            processed_image = preprocess_image(image_data, target_size)
                            print(os.path.join(json_dir, expname, well_name, pos_name, filename),  '  alive=',alive)
                            if len(processed_image)!=0:
                                sequence.append(processed_image)
                                label.append(1 if alive == True else 0)
                                frame.append(int(filename.replace("frame","").split("-")[1]))

                sorted_lists = sorted(zip(frame, sequence, label)) 
                frame_sorted, sequence_sorted, label_sorted = zip(*sorted_lists) 

                print(frame_sorted)
                print(sequence_sorted)
                print(label_sorted)
                
                sys.exit(3)
                sequences.append(sequence)
                labels.append(labels)

                
    return np.array(sequences), np.array(labels)



images, labels = load_and_preprocess_images(json_dir)
images = np.expand_dims(images, axis=-1)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


input_shape = (None, 150, 150, 1)  # None allows for variable sequence lengths

model = models.Sequential([
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.5),
    layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
])




model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
#history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
#                    epochs=50,
#                    validation_data=(x_val, y_val),
#                    callbacks=[reduce_lr, early_stopping])

history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    callbacks=[reduce_lr, early_stopping])




#model.save('cell_classifier_model.h5')
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