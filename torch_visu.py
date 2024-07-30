import torch
import json, os, random, time

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

def visualize_predictions(images, predictions, boxes):
    print('images, predictions, boxes ',len(images),'  ', len(predictions),'  ', len(boxes))
    for idx, (image, pred, box) in enumerate(zip(images, predictions, boxes)):

        fig, ax = plt.subplots(1)
        img = image.squeeze(0).squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray')
        #for p in pred[0]['boxes']:
        for p, score in zip(pred['boxes'], pred['scores']):
            print('=================',p, score)
            x_min, y_min, x_max, y_max = p.cpu().numpy()
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x_min, y_min, f'{score:.2f}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        for b in box:
            rect = patches.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], linewidth=1, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        plt.show()





def main():

    # Load the model
    start_time = time.time()
    model_path = 'cell_detection_model.pth'
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load the model
    model = load_model(model_path, num_classes, device)

    end_time = time.time()
    print(f"Loading model took {end_time - start_time:.4f} seconds")


    json_path = r'D:\single_cells\training_cell_detection\wscepfl0080\wscepfl0080_well2\wscepfl0080_xy41\frame0.json'
    base_path = r'D:\single_cells\training_cell_detection\wscepfl0080'
    base_path = r'D:\single_cells\training_cell_detection\wscepfl0060\wscepfl0060_well1'
    json_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                
                if 'xy04' not in root:continue
                #if 'xy40' not in root:continue
                json_files.append(os.path.join(root, file))


    processed_images=[]
    boxes=[]
    #random.shuffle(json_files)
    for idx in range(len(json_files)):
        print(json_files[idx])
        json_path=json_files[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)
        image_array = np.array(data['data'], dtype=np.float32)  # Convert to float32 to avoid overflow
        boxes.append( [ann['bbox'] for ann in data['annotations']])

        # Preprocess the image
        image = preprocess_image(image_array).to(device)
        processed_images.append(image)

    # Convert list of images to a batch tensor
    batch_images = torch.stack(processed_images).to('cuda')
    # Make predictions
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image)
    end_time = time.time()
    print(f"Inference on batch size {len(batch_images)} took {end_time - start_time:.4f} seconds")

    # Visualize the predictions
    visualize_predictions(processed_images, predictions, boxes)

if __name__ == "__main__":
    main()
