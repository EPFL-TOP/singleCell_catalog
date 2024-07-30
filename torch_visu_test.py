import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import time, os, json

def load_model(model_path, device):
    # Assuming model is saved with torch.save and the architecture is known
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model

def load_images(image_paths, device):
    images = []
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert to tensor and normalize if needed
    ])
    for path in image_paths:

        with open(path, 'r') as f:
            data = json.load(f)
        image = np.array(data['data'], dtype=np.float32)  # Convert to float32 to avoid overflow

        #image = np.array(path[])  # Load 2D NumPy array from file
        image = transform(image).unsqueeze(0)  # Add batch dimension
        images.append(image.to(device))
    return images

def visualize_predictions(images, predictions):
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        fig, ax = plt.subplots(1)
        ax.imshow(image.squeeze(0).cpu().numpy(), cmap='gray')
        if prediction is not None:
            for box in prediction['boxes']:
                x_min, y_min, x_max, y_max = box
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, color='red')
                ax.add_patch(rect)
        plt.show()

def infer_images(image_paths, model_path, device, batch_size=1):
    model = load_model(model_path, device)
    images = load_images(image_paths, device)
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_images = [image.squeeze(0) for image in batch_images]  # Remove batch dimension
            
            # Ensure images are in the correct format for the model
            if batch_images:
                batch_images = torch.stack(batch_images)  # Convert list of tensors to a single tensor
                predictions = model(batch_images)
            
                # Convert predictions to CPU and format them for visualization
                predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            
                # Visualize predictions for the first batch
                if i == 0:
                    visualize_predictions(batch_images, predictions)
                    
    end_time = time.time()
    
    print(f"Inference on {len(images)} images took {end_time - start_time:.2f} seconds")

def main():
    base_path = r'D:\single_cells\training_cell_detection\wscepfl0060\wscepfl0060_well1'
    image_paths = []  # Update paths
    image_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                
                if 'xy04' not in root:continue
                #if 'xy40' not in root:continue
                image_paths.append(os.path.join(root, file))

    model_path = 'cell_detection_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_images(image_paths, model_path, device, batch_size=4)

if __name__ == "__main__":
    main()
