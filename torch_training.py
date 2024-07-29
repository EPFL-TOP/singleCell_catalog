import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CellDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path  # Base directory containing all experimental conditions
        self.transform = transform
        self.json_files = self._collect_json_files()

    def _collect_json_files(self):
        json_files = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        image_array = np.array(data['data'], dtype=np.float32)  # Convert to float32 to avoid overflow
        boxes = [ann['bbox'] for ann in data['annotations']]
        
        # Convert bounding boxes from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all cells belong to one class

        if self.transform:
            image = self.transform(image_array)
        else:
            image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add channel dimension

        target = {
            'boxes': boxes,
            'labels': labels
        }
        return image, target

class ToTensorNormalize:
    def __call__(self, image):
        # If input is a NumPy array
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        else:  # Handle PIL Image (if applicable)
            image = transforms.functional.pil_to_tensor(image).float()
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        return image

transform = transforms.Compose([
    ToTensorNormalize(),
    transforms.Resize((512, 512))
])

base_path = r'D:\single_cells\training_cell_detection'

train_dataset = CellDataset(base_path=base_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))






import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2  # 1 class (cell) + background
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=1e-4)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
