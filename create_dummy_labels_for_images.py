# create_dummy_labels_for_images.py

import os
import numpy as np

# Folder containing images
images_folder = 'images/'

# Generate .txt labels for each image
for filename in os.listdir(images_folder):
    if filename.endswith('.png'):
        label_filename = filename.replace('.png', '.txt')
        label_path = os.path.join(images_folder, label_filename)

        # Random (x, y, z, score) values between 0 and 1
        label = np.random.uniform(0, 1, size=4)

        with open(label_path, 'w') as f:
            f.write(' '.join(map(str, label)))

print(f"Generated dummy labels for all images in {images_folder}")
