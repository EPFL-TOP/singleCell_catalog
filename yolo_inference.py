

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path



def preprocess_image(image_array):
    # Normalize the image
    print(image_array.shape)
    image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    # Expand dimensions to match the expected input shape (1, 512, 512) -> (1, 1, 512, 512) -> (1, 3, 512, 512)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print(image.shape)
    #image = np.repeat(image, 3, axis=1)    # Repeat the single channel to create 3 channels
    print(image.shape)
    print(image.dtype)    # Ensure the image is 2D and convert to 3 channels
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
        if '.json' in image_path:
            with open(image_path, 'r') as f:
                data = json.load(f)
            image_array = np.array(data['data'], dtype=np.float32)
            image_array = preprocess_image(image_array)

            # Inference
            results = model.predict(image_array)

        else:
            results = model.predict(image_path)

        # Extracting the predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                predictions.append({
                    'box': box,
                    'confidence': conf
                })

        # Visualize the results
        #visualize_predictions(image_array[0, 0], predictions)
        # Visualize the results
        visualize_predictions(image_path, predictions)

# Paths to your test images
# Paths to your test images
image_paths = [r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame0.json', 
               r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame1.json']
image_paths = [r'D:\single_cells\training_cell_detection_YOLO\images\val\fna-bleb001_xy222_e5d5e17e-4dd0-11ef-ab07-ac1f6bbc3550.png']
model_path = r'C:\Users\helsens\software\singleCell_catalog\runs\detect\train\weights\best.pt'
infer_images(image_paths, model_path)
