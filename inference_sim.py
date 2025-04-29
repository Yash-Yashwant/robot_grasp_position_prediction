#now will be using the trained CNN model and try to simulate it using pybullet.

import torch
from torchvision import transforms
from cnn_model import GraspCNN
from sim_utils import setup_simulation, spawn_random_cube, setup_camera, capture_rgb_image, step_simulation, disconnect
import pybullet as p
import numpy as np
from PIL import Image
import time

# Setup device
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Load trained model
model = GraspCNN().to(device)
model.load_state_dict(torch.load('grasp_model.pth', map_location=device))
model.eval()

# Setup simulation
robot_id = setup_simulation()
view_matrix, proj_matrix, width, height = setup_camera()

# Spawn random cube
cube_id, (x_gt, y_gt, z_gt) = spawn_random_cube()
print(f"Cube ground-truth position: {x_gt}, {y_gt}, {z_gt}")

# Capture image
capture_rgb_image(view_matrix, proj_matrix, width, height, 'inference_sample.png')
img = Image.open('inference_sample.png').convert('RGB')

# Transform image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

img_tensor = transform(img).unsqueeze(0).to(device)

# Predict grasp position
with torch.no_grad():
    output = model(img_tensor)
    pred_xyz = output[:, :3]
    x_pred, y_pred, z_pred = pred_xyz[0].cpu().numpy()

print(f"Predicted grasp position: {x_pred}, {y_pred}, {z_pred}")

# Use IK to move to predicted grasp location
target_position = [x_pred, y_pred, z_pred + 0.05]  # Hover slightly above
target_orientation = p.getQuaternionFromEuler([0, np.pi, 0])

joint_angles = p.calculateInverseKinematics(robot_id, 11, target_position, target_orientation)

# Move robot joints smoothly
for _ in range(240):
    for joint_idx in range(7):
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, joint_angles[joint_idx], force=500)
    p.stepSimulation()
    time.sleep(1. / 240)

# Lower down to cube
target_position[2] -= 0.05  # Move down by 5cm
joint_angles = p.calculateInverseKinematics(robot_id, 11, target_position, target_orientation)

for _ in range(240):
    for joint_idx in range(7):
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, joint_angles[joint_idx], force=500)
    p.stepSimulation()
    time.sleep(1. / 240)

# (Optional) Close gripper here if you have gripper code
# For now just simulate motion

# Lift the object up
target_position[2] += 0.2  # Lift up
joint_angles = p.calculateInverseKinematics(robot_id, 11, target_position, target_orientation)

for _ in range(480):
    for joint_idx in range(7):
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, joint_angles[joint_idx], force=500)
    p.stepSimulation()
    time.sleep(1. / 240)

print("Grasp attempt completed.")
step_simulation(steps=1000)

disconnect()
