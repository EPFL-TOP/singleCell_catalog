

import torch
import torchvision
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform

#from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt


# Mapping subfolder names to labels
cell_types = {"normal": 1, "elongated": 2, "dead": 3, "flat": 4}
cell_types = {"normal": 1, "dead": 2}

def get_transform():
    return ToTensor()


class CellDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_files = []
        self.annotation_files = []
        self.labels = []
        for cell_type, label in cell_types.items():
            cell_type_dir = os.path.join(root_dir, cell_type)
            if os.path.isdir(cell_type_dir):
                files = list(sorted(os.listdir(cell_type_dir)))
                for file in files:
                    file = os.path.join(cell_type_dir, file)
                    if '_annotation' not in file: continue
                    data={}
                    with open(file) as f:
                        data = json.load(f)
                        valid=True
                        try:
                            valid=data["valid"]
                        except KeyError:
                            valid=True
                    if not valid: continue
                    self.data_files.append(data["image_json"])
                    self.annotation_files.append(file)
                    self.labels.append(label)


    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        json_path = self.data_files[idx]
        with open(json_path) as f:
            data = json.load(f)
        
        img = np.array(data["data"])
        img = np.array(img, dtype=np.float32) / 65535.0  # Normalize to [0, 1] based on int16 max
        print(f"Image shape 1: {img.shape}")  # Should be [3, H, W]

        img = np.expand_dims(img, axis=0)  # Add channel dimension
        print(f"Image shape 2: {img.shape}")  # Should be [3, H, W]


        #img = np.expand_dims(img, axis=0)  # Make it (1, H, W)
        img = np.repeat(img, 3, axis=0)   # Convert to (3, H, W)
        print(f"Image shape 3: {img.shape}")  # Should be [3, H, W]

        img = np.transpose(img, (0, 1, 2))

        json_path = self.annotation_files[idx]
        with open(json_path) as f:
            data = json.load(f)

        boxes = [[data["bbox"][0], data["bbox"][2], data["bbox"][1], data["bbox"][3]]]
        #boxes = [[x_min, y_min, x_max, y_max] for x_min, x_max, y_min, y_max in boxes]
#"bbox":[cellroi.min_col, cellroi.max_col, cellroi.min_row, cellroi.max_row],
        label = self.labels[idx]
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.full((len(boxes),), label, dtype=torch.int64)  # All boxes have the same label

        if self.transforms:
            img = self.transforms(img)
        print(f"Image shape 4: {img.shape}")  # Should be [3, H, W]
    
        return img, target
    

def collate_fn(batch):
    return tuple(zip(*batch))



# Create a transform specifically for grayscale images
#class CustomRCNNTransform(GeneralizedRCNNTransform):
#    def normalize(self, image):
#        mean = torch.tensor([0.5], device=image.device)  # Mean for grayscale
#        std = torch.tensor([0.5], device=image.device)   # Std for grayscale
#        return (image - mean[:, None, None]) / std[:, None, None]




def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    return running_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
    
    return running_loss / len(data_loader)


if __name__ == "__main__":
    dataset = CellDataset(root_dir=r'D:\single_cells\training_cell_detection_categories_new', transforms=get_transform())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    #val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"Number of training images: {train_size}")
    print(f"Number of validation images: {val_size}")


    #model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT', pretrained=True)
    # Use this custom transform in your model initialization
    #model.transform = CustomRCNNTransform(min_size=512, max_size=512, image_mean=[0.5], image_std=[0.5])


    num_classes = 3  # 1 class (cell) + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate(model, val_loader, device)
        
        lr_scheduler.step()

        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
