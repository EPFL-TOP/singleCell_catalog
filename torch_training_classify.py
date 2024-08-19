import torch
from torch.utils.data import Dataset, DataLoader
import cv2  # OpenCV for image processing
import os

class CellDataset(Dataset):
    def __init__(self, image_paths, bounding_boxes, labels, transform=None):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes  # list of (x, y, w, h) tuples
        self.labels = labels  # list of labels: "normal", "dead", etc.
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Extract the bounding box and crop the cell
        x, y, w, h = self.bounding_boxes[idx]
        cell_image = image[y:y+h, x:x+w]
        
        # Resize the image to a consistent size (e.g., 64x64)
        cell_image = cv2.resize(cell_image, (64, 64))
        
        # Apply any transformations (normalization, etc.)
        if self.transform:
            cell_image = self.transform(cell_image)
        
        # Get label and convert to tensor
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return cell_image, label