import tensorflow as tf
import json, os
import numpy as np
json_dir='/data/singleCell_training/wscepfl0080/wscepfl0080_well1/'
json_dir='/data/singleCell_training/ppf003/ppf003_well1'
json_dir='/data/singleCell_training/bleb001/bleb001_well1/fna-bleb001_xy001'
json_dir='/data/singleCell_training/wscepfl0102/wscepfl0102_well1/wscepfl0102_xy03/'
json_dir='/data/singleCell_training/wscepfl0102/wscepfl0102_well1/wscepfl0102_xy03/'
json_dir='/data/singleCell_training/bleb002/bleb002_well1/fna-bleb002_xy005/'
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image, target_size = (150, 150)):
    # Calculate padding
    image_data = np.array(image, dtype=np.int16)
    delta_w = target_size[1] - image_data.shape[1]
    delta_h = target_size[0] - image_data.shape[0]
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
    padded_image = np.expand_dims(padded_image, axis=-1)

    return padded_image



def load_and_preprocess_image(json_file, target_size=(150, 150)):

    with open(json_file, 'r') as f:
        data = json.load(f)
        image_data = data['image_bf']
        processed_image = preprocess_image(image_data, target_size)
    return processed_image


def load_and_preprocess_images(json_dir, target_size=(150, 150)):
    images = []
    filenames = []

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename), 'r') as f:
                data = json.load(f)
                image_data = data['image_bf']
                processed_image = preprocess_image(image_data, target_size)
                images.append(processed_image)
                filenames.append(filename)
    return np.array(images), filenames


# Load the saved model
model = load_model('cell_classifier_model.keras')
#model = load_model('cell_classifier_model.h5')

new_images, filenames = load_and_preprocess_images(json_dir)
# Add batch dimension if not already added
if len(new_images.shape) == 3:
    new_images = np.expand_dims(new_images, axis=-1)

# Predict the classes for the batch of images
predictions = model.predict(new_images)

# Interpret the predictions
predicted_classes = ['ALIVE' if pred > 0.5 else 'DEAD' for pred in predictions]

# Print the results
for filename, predicted_class, pred in zip(filenames, predicted_classes, predictions):
    print(f'File: {filename}, Predicted class: {predicted_class}   weight={pred}')


import sys
sys.exit(3)

# Path to a new image JSON file
for dirpath, dirnames, filenames in os.walk(json_dir):
    for filename in filenames:
        if filename.endswith(".json"):
            print(dirpath,'  ',filename)

            new_image_json = os.path.join(dirpath, filename)

            # Load and preprocess the new image
            new_image = load_and_preprocess_image(new_image_json)

            # Add batch dimension
            new_image = np.expand_dims(new_image, axis=0)

            # Predict the class
            prediction = model.predict(new_image)

            # Interpret the prediction
            predicted_class = 'ALIVE' if prediction[0] > 0.5 else 'DEAD'
            print(f'Predicted class: {predicted_class}  weight={prediction[0]}')