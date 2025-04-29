import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from cnn_model import GraspCNN
from dataloader import GraspDataset



# Settings
MODEL_PATH = 'saved_models/model.pth'  # <-- Adjust if different
TEST_DATASET_PATH = 'images/'        # <-- Adjust if different
BATCH_SIZE = 32
SUCCESS_THRESHOLD_CM = 5.0  # Define what counts as "success"

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraspCNN()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Loaded trained model.")
else:
    print("No trained model found. Using randomly initialized model for evaluation.")

model.to(device)
model.eval()

# Load test dataset
test_dataset = GraspDataset(TEST_DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation loop
total_samples = 0
total_successes = 0
placement_errors = []

for images, true_labels in test_loader:
    images = images.to(device)
    true_labels = true_labels.to(device)

    with torch.no_grad():
        preds = model(images)

    preds = preds.squeeze()
    true_labels = true_labels.squeeze()

    # Only compare first 3 dimensions (x, y, z)
    preds_xyz = preds[:, :3]
    true_xyz = true_labels[:, :3]

    # Calculate Euclidean distance error
    error = torch.norm(preds_xyz - true_xyz, dim=1)  # shape (batch_size,)
    placement_errors.extend(error.cpu().numpy())

    # Success = error < threshold
    successes = (error < SUCCESS_THRESHOLD_CM).sum().item()
    total_successes += successes
    total_samples += images.size(0)

# Results
mean_error = np.mean(placement_errors)
success_rate = (total_successes / total_samples) * 100

print(f"Mean Placement Error: {mean_error:.2f} cm")
print(f"Success Rate: {success_rate:.2f}%")
