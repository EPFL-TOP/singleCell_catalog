import torch
import json

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

def visualize_predictions(image, predictions):
    fig, ax = plt.subplots(1)
    img = image.squeeze(0).squeeze(0).cpu().numpy()
    ax.imshow(img, cmap='gray')
    for box in predictions[0]['boxes']:
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()

def main():
    model_path = 'cell_detection_model.pth'
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model
    model = load_model(model_path, num_classes, device)

    # Example 2D array image
    #image_array = np.random.randint(0, 256, (512, 512), dtype=np.uint8)  # Replace with your actual 2D numpy array
    json_path = r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well2\wscepfl0080_xy41\frame0.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_array = np.array(data['data'], dtype=np.float32)  # Convert to float32 to avoid overflow


    # Preprocess the image
    image = preprocess_image(image_array).to(device)

    # Make predictions
    with torch.no_grad():
        predictions = model(image)

    # Visualize the predictions
    visualize_predictions(image.cpu(), predictions)

if __name__ == "__main__":
    main()
