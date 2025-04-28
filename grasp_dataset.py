# grasp_dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class GraspDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Labels: x, y, z, score
        label = torch.tensor([row['x'], row['y'], row['z'], row['label']], dtype=torch.float32)

        return image, label
