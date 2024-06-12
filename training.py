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

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Stop training when a monitored metric has stopped improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Define the paths
json_dir = '/data/singleCell_training/'

# Initialize lists to store images and labels
images = []
labels = []


# Target image size (height, width)
target_size = (150, 150)

def pad_image(image, target_size):
    # Calculate padding
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

    return padded_image

# Function to load JSON data and convert to numpy arrays
def load_json_data(json_dir):

    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                #print(dirpath,'  ',filename)
                with open(os.path.join(dirpath, filename), 'r') as f:
                    data = json.load(f)
                    image_data = np.array(data['image_bf'], dtype=np.int16)
                    label = data['alive']
                    
                    # Append the image and label to lists
                    #images.append(image_data)
                    # Pad the image
                    padded_image = pad_image(image_data, target_size)

                    # Normalize the image data to range [0, 1]
                    padded_image = padded_image / np.max(padded_image)
                    
                    # Expand dimensions to match expected input shape (height, width, channels)
                    padded_image = np.expand_dims(padded_image, axis=-1)

                    # Append the image and label to lists
                    images.append(padded_image)
                    labels.append(1 if label == True else 0)  # Assuming 'dead_cell' is labeled as 1, 'live_cell' as 0

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Load data
load_json_data(json_dir)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize image data to range [0, 1]
images = images / np.max(images)

# Expand dimensions to match expected input shape (samples, height, width, channels)
images = np.expand_dims(images, axis=-1)


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = np.expand_dims(X_train, axis=-1)
datagen.fit(X_train)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(images.shape[1], images.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    #X_train, y_train,
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    batch_size=20,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr, early_stopping]
)

#model.save('cell_classifier_model.h5')
model.save('cell_classifier_model.keras')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.savefig('training_acc.png')

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('training_loss.png')