import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json
def preprocess_image(image_array):
    # Normalize the image
    image = (image_array - image_array.min()) / (image_array.max() - image_array.min())
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
    model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path)

    for image_path in image_paths:
        with open(image_path, 'r') as f:
            data = json.load(f)
        image_array = np.array(data['data'], dtype=np.float32)
        image_array = preprocess_image(image_array)

        # Inference
        results = model(image_array)

        # Extracting the predictions
        predictions = []
        for result in results.xyxy[0]:
            x_min, y_min, x_max, y_max, conf, cls = result.cpu().numpy()
            predictions.append({
                'box': [x_min, y_min, x_max, y_max],
                'confidence': conf
            })

        # Visualize the results
        visualize_predictions(image_array, predictions)

# Paths to your test images
image_paths = ['D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame0.json', 
               'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well1\wscepfl0080_xy01\frame1.json']
model_path = 'runs/detect/train/weights/best.pt'
infer_images(image_paths, model_path)
