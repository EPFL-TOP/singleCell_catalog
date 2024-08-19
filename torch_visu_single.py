import torch
import json, os, random, datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

class ToTensorNormalize:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            image = F.pil_to_tensor(image).float()
        
        image = (image - image.min()) / (image.max() - image.min())
        return image

def load_model(model_path, num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_array):
    transform = ToTensorNormalize()
    image = transform(image_array)
    return image.unsqueeze(0)  # Add batch dimension

def visualize_predictions(image, predictions, boxes):
    fig, ax = plt.subplots(1)
    img = image.squeeze(0).squeeze(0).cpu().numpy()
    print(predictions)
    ax.imshow(img, cmap='gray')
    for box in predictions[0]['boxes']:
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    for box in boxes:
        rect = patches.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def main():
    model_path = 'cell_detection_model.pth'
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model
    model_gpu = load_model(model_path, num_classes, torch.device('cuda'))
    model_cpu = load_model(model_path, num_classes, torch.device('cpu'))


    base_path=r'D:\single_cells\training_cell_detection_categories\valid'

    json_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('_annotation.json'):
                with open(file) as f:
                    data = json.load(f)
                    valid=True
                    try:
                        valid=data["valid"]
                    except KeyError:
                        valid=True
                if not valid: continue
                json_files.append(os.path.join(root, file))

    random.shuffle(json_files)
    nimages=100
    for idx in range(len(json_files)):
        print(json_files[idx])
        if idx>=nimages:break
        json_path=json_files[idx]
        with open(json_path, 'r') as f:
            data_annotation = json.load(f)
        with open(data_annotation['image_json'], 'r') as f:
            data = json.load(f)
        image_array = np.array(data['data'], dtype=np.float32)
        boxes = [data_annotation['bbox']]

        # Preprocess the image
        image = preprocess_image(image_array).to(device)

        # Make predictions
        with torch.no_grad():
            start=datetime.datetime.now()
            predictions = model_gpu(image)
            end=datetime.datetime.now()
            print('took: ',end-start)
        # Visualize the predictions
        visualize_predictions(image.cpu(), predictions, boxes)

if __name__ == "__main__":
    main()
