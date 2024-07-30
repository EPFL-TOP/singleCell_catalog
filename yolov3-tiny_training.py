import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# Assuming YOLOv3Tiny is defined and available
from models import YOLOv3Tiny  # Import your YOLOv3Tiny model implementation


class CellDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])
        
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                boxes.append([class_id, x_center, y_center, width, height])
        
        return image, torch.tensor(boxes)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # Converts the image to a tensor and normalizes pixel values to [0, 1]
])



train_dataset = CellDataset(r'D:\single_cells\training_cell_detection_YOLO\images\train', r'D:\single_cells\training_cell_detection_YOLO\labels\train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

val_dataset = CellDataset(r'D:\single_cells\training_cell_detection_YOLO\images\val', r'D:\single_cells\training_cell_detection_YOLO\labels\val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


model = YOLOv3Tiny(num_classes=1)  # Adjust according to your model's implementation
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()  # Adjust according to your needs


num_epochs = 10  # Adjust the number of epochs as needed

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        
        # Assuming the outputs and labels are in the right format for your loss function
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'cell_detection_yolov3_tiny.pth')
