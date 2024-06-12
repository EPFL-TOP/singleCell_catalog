import tensorflow as tf
import json, os
import numpy as np
json_dir='/data/singleCell_training/wscepfl0080/wscepfl0080_well1/'
json_dir='/data/singleCell_training/ppf003/ppf003_well1'
json_dir='/data/singleCell_training/bleb001/bleb001_well1/'

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



# Load the saved model
model = load_model('cell_classifier_model.keras')
#model = load_model('cell_classifier_model.h5')

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