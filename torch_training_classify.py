import torch
from torch.utils.data import Dataset, DataLoader
import cv2  # OpenCV for image processing
import os, json
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


labels_map = {'normal':1, 'dead':2, 'flat':3, 'elongated':4, 'dividing':5}

class CellDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.json_files = self._collect_json_files()

    def _collect_json_files(self):
        json_files = []
        for root, _, files in os.walk(self.image_paths):
            for file in files:
                if file.endswith('_annotation.json'):

                    with open(os.path.join(root, file)) as f:
                        data = json.load(f)
                        valid=True
                        try:
                            valid=data["valid_label"]
                        except KeyError:
                            valid=True#to be changed to False when validation done
                    if not valid: continue

                    json_files.append(os.path.join(root, file))
        return json_files

        self.bounding_boxes = bounding_boxes  # list of (x, y, w, h) tuples
        self.labels = labels  # list of labels: "normal", "dead", etc.

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):

        json_path = self.json_files[idx]
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        with open(annotations['image_json'], 'r') as f:
            data = json.load(f)

        image_array = np.array(data['data_cropped'], dtype=np.float32)  # Convert to float32 to avoid overflow

        # Get label and convert to tensor
        label = int(annotations["label"])
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image_array)
        else:
            image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add channel dimension

        return image, label




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
  #  transforms.Resize((512, 512))
])


    


class CellClassifier(nn.Module):
    def __init__(self):
        super(CellClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input is 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust size according to your input dimensions
        self.fc2 = nn.Linear(512, 4)  # 4 classes: "normal", "dead", "flat", "elongated"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



image_paths=r'D:\single_cells\training_cell_detection_categories\train'
model_save_path = 'cell_labels_model.pth'

# Load your dataset
train_dataset = CellDataset(image_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = CellClassifier()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device: ',device)
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #inputs = inputs.unsqueeze(1)  # Add a channel dimension for grayscale

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            print(f'[{epoch + 1}, {i}] loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    #torch.save(model.state_dict(), model_save_path)
    torch.save(checkpoint, model_save_path)
    print(f'Model saved to {model_save_path} after epoch {epoch + 1 }')


print('Finished Training')




