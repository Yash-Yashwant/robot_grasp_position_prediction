# Robot Grasp Position Prediction Using Convolutional Neural Networks

**Contributors:**  
- Nikhil Sawane  
- Yashwant Gandham

---

## Project Overview

This project builds a pipeline to predict robot grasp placement (x, y, z) coordinates directly from RGB camera images using a Convolutional Neural Network (CNN). The simulation environment was created using PyBullet, and evaluation was performed on synthetic images with dummy labels. Although no training was performed, the project verifies the simulation, model, and evaluation flow.

---

## Methodology

- **Simulation Environment:** PyBullet with Franka Panda robot arm.
- **Dataset:** 100 synthetic RGB images, each with randomly generated (x, y, z, score) labels.
- **Model Architecture:** A lightweight CNN with 3 convolutional layers followed by 3 fully connected layers.
- **Evaluation Metrics:** Mean placement error (cm) and success rate (% of predictions within a 5 cm threshold).

---

## How to Run

1. Clone the repository.
2. Set up a Python environment and install dependencies.
3. (Optional) Generate dummy labels by running `create_dummy_labels_for_images.py`.
4. Run `evaluate.py` to evaluate the model.
5. (Optional) Run `plot_errors.py` to generate the placement error histogram.

---

## Results

- **Mean Placement Error:** ~1.00 cm
- **Success Rate:** 100% (with 5 cm error threshold)

Note: These results are based on a randomly initialized CNN and dummy labels.

---

## Future Work

- Collect real grasp placement labels.
- Train the CNN to minimize placement errors.
- Test in dynamic environments and real-world robotic systems.

---

## Acknowledgments

- CSCI 5922 - Neural Nets and Deep Learning, Spring 2025
- Thanks to the instructors and TAs for their guidance.

---
