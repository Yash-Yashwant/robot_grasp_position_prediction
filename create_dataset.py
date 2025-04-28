# create_dataset.py

import os
import csv
from sim_utils import setup_simulation, spawn_random_cube, setup_camera, capture_rgb_image, step_simulation, disconnect
import pybullet as p
# Settings
num_samples = 100
image_folder = "images"
csv_file = "grasp_data.csv"

# Create image directory if it doesn't exist
os.makedirs(image_folder, exist_ok=True)

# Open CSV file for writing
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "x", "y", "z", "label"])  # label is -1 for now

    # Start PyBullet simulation
    robot_id = setup_simulation()
    view_matrix, proj_matrix, width, height = setup_camera()

    for i in range(1, num_samples + 1):
        # Spawn cube
        cube_id, (x, y, z) = spawn_random_cube()

        # Capture image
        image_name = f"grasp_{i:03d}.png"
        save_path = os.path.join(image_folder, image_name)
        capture_rgb_image(view_matrix, proj_matrix, width, height, save_path)

        # Save row in CSV (label = -1 as placeholder)
        writer.writerow([image_name, x, y, z, -1])

        # Step the simulation to render the scene
        step_simulation(steps=100)

        # Remove the cube before next iteration
        p.removeBody(cube_id)

    disconnect()

print(f"Dataset creation complete: {num_samples} samples saved to '{csv_file}'")