import torch, torchvision
import json, os, random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def evaluate_model(model, images, device):
    model.eval()
    with torch.no_grad():
        #images = [torch.tensor(image).unsqueeze(0).to(device) for image in images]  # Add batch dimension
        outputs = model(images)
    return outputs

# Example usage
# Load your model


model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT', pretrained=True)
num_classes = 3  # 1 class (cell) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("model_epoch_10.pth"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)




# Example evaluation
test_images = [np.random.rand(1, 512, 512).astype(np.float32)]  # Replace with your test 2D arrays
test_images = [np.random.rand(1, 512, 512).astype(np.float32)]  # Replace with your test 2D arrays
image=r'D:\single_cells\training_cell_detection_categories_new\normal\bleb001_xy049_frame17_cell0.json'
with open(image, 'r') as f:
    data = json.load(f)
image_array = np.array(data['data'], dtype=np.float32)  
img = np.array(image_array, dtype=np.float32) / 65535.0  # Normalize to [0, 1] based on int16 max
img = np.expand_dims(img, axis=0)  # Add channel dimension
img = np.repeat(img, 3, axis=0)   # Convert to (3, H, W)

#outputs = evaluate_model(model, test_images, device)
outputs = evaluate_model(model, img, device)
for output in outputs:
    print(output['boxes'], output['scores'])