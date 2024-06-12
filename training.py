import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the paths
json_dir = '/data/singleCell_training/'

# Initialize lists to store images and labels
images = []
labels = []

# Function to load JSON data and convert to numpy arrays
def load_json_data(json_dir):

    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                print(dirpath,'  ',filename)
                with open(os.path.join(dirpath, filename), 'r') as f:
                    data = json.load(f)
                    image_data = np.array(data['image_bf'], dtype=np.int16)
                    label = data['alive']
                    
                    # Append the image and label to lists
                    images.append(image_data)
                    labels.append(1 if label == True else 0)  # Assuming 'dead_cell' is labeled as 1, 'live_cell' as 0

print(labels)
print(images)

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
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=20,
    validation_data=(X_val, y_val)
)




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

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()