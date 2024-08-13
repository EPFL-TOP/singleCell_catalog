import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor

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
        self.labels = []
        for cell_type, label in cell_types.items():
            cell_type_dir = os.path.join(root_dir, cell_type)
            if os.path.isdir(cell_type_dir):
                files = list(sorted(os.listdir(cell_type_dir)))
                for file in files:
                    file = os.path.join(cell_type_dir, file)
                    if '_annotation' not in file: continue
                    print('processing ',file)
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
                self.labels.append(label)



    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        json_path = self.data_files[idx]
        with open(json_path) as f:
            data = json.load(f)
        
        img = np.array(data["data"])
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        
        boxes = [ann["bbox"] for ann in data["annotations"]]
        label = self.labels[idx]
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.full((len(boxes),), label, dtype=torch.int64)  # All boxes have the same label

        if self.transforms:
            img = self.transforms(img)
        
        return img, target
    

dataset = CellDataset(root_dir=r'D:\single_cells\training_cell_detection_categories_new', transforms=get_transform())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

print(f"Number of training images: {train_size}")
print(f"Number of validation images: {val_size}")