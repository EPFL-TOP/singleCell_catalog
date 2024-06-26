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
import mva_utils as mva_utils

# Define the paths
json_dir = '/mnt/sdc1/data/singleCell_training/'
if not os.path.exists('/mnt/sdc1/data/singleCell_training'):
    json_dir = '/data/singleCell_training/'


# Target image size (height, width)
target_size = (150, 150)
use_tl      = False
complex     = False
model       = None
callbacks   = None
model_name  = 'cell_classifier_model.keras'
plot_name   = 'training_acc_loss.png'
nbatch      = 32
property    = {'main':'alive'}
property    = {'main':'oscillating', 'conditions':{'alive':True}}

images, labels = mva_utils.load_and_preprocess_images(json_dir, target_size=target_size, property=property)
if not use_tl: images = np.expand_dims(images, axis=-1)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
if use_tl:
    print(x_train.shape)
    x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
    x_val = np.repeat(x_val[..., np.newaxis], 3, -1)
    print(x_train.shape)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(mva_utils.scheduler)


#No TL
#______________________________________________________________________
if not use_tl:

    if complex:
        model = models.Sequential([
        mva_utils.data_augmentation_complex,
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

        model_name = 'cell_classifier_model_complex_{}.keras'.format(property['main'])
        plot_name  = 'training_acc_loss_complex_{}.png'.format(property['main'])


    else:   
        model = models.Sequential([
        mva_utils.data_augmentation_simple,

        #layers.Input(shape=(150, 150, 1)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(512, activation='relu'),

        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

        model_name = 'cell_classifier_model_simple_{}.keras'.format(property['main'])
        plot_name  = 'training_acc_loss_simple_{}.png'.format(property['main'])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        #lr_scheduler,
]

#when TL
#______________________________________________________________________
if use_tl:

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    base_model_tl = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model_tl = tf.keras.applications.EfficientNetB0(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model_tl = tf.keras.applications.VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    nbatch = 24

    model = models.Sequential([
    mva_utils.data_augmentation_simple,
    #layers.Input(shape=(150, 150, 3)),  # Convert grayscale to RGB if needed

    base_model_tl,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    #history = model.fit(x_train, y_train,
    #                    epochs=100,
    #                    batch_size=0,
    #                    validation_data=(x_val, y_val),
    #                    callbacks=callbacks)



    base_model_tl.trainable = True
    for layer in base_model_tl.layers[:-20]:  # Unfreeze the last 20 layers
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ]


    model_name = 'cell_classifier_model_tl_{}.keras'.format(property['main'])
    plot_name  = 'training_acc_loss_tl_{}.png'.format(property['main'])

history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=nbatch,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks)

model.save(model_name)

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