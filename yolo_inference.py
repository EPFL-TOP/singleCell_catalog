

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path





def preprocess_image(image_array):
    print(image_array)
    print(image_array.shape)
    print(image_array.dtype)    # Ensure the image is 2D and convert to 3 channels
    if len(image_array.shape) == 2:
        image = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 1:
        image = np.concatenate((image_array, image_array, image_array), axis=-1)
    else:
        image = image_array

    # Resize to (512, 512) if necessary
    if image.shape[:2] != (512, 512):
        image = cv2.resize(image, (512, 512))

    # Normalize the image to range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Convert to float32
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def visualize_predictions(image_array, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image_array, cmap='gray')

    for pred in predictions:
        box = pred['box']
        conf = pred['confidence']
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, f'{conf:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

def infer_images(image_paths, model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    for image_path in image_paths:
        with open(image_path, 'r') as f:
            data = json.load(f)
        image_array = np.array(data['data'], dtype=np.float32)
        image_array = preprocess_image(image_array)

        # Inference
        results = model.predict(image_array)

        # Extracting the predictions
        predictions = []
        for result in results:
            for r in result.boxes:
                x_min, y_min, x_max, y_max, conf = r[:5]
                predictions.append({
                    'box': [x_min, y_min, x_max, y_max],
                    'confidence': conf
                })

        # Visualize the results
        visualize_predictions(image_array, predictions)

# Paths to your test images
# Paths to your test images
image_paths = [r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame0.json', 
               r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame1.json']
model_path = r'C:\Users\helsens\software\singleCell_catalog\runs\detect\train\weights\best.pt'
infer_images(image_paths, model_path)
