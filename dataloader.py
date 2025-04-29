# dataloader.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GraspDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Assuming corresponding labels are stored as .txt files with same name
        self.label_files = [f.replace('.png', '.txt').replace('.jpg', '.txt') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor([float(x) for x in open(label_path).read().strip().split()], dtype=torch.float)

        return image, label
