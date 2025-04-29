# plot_errors.py

import matplotlib.pyplot as plt
import numpy as np

# Manually put your evaluation results here
placement_errors = np.random.normal(1.0, 0.2, 100)  # Simulating errors around 1.0 cm

# Plot histogram
plt.figure(figsize=(8,6))
plt.hist(placement_errors, bins=20, edgecolor='black')
plt.title('Distribution of Placement Errors')
plt.xlabel('Placement Error (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('error_histogram.png')
plt.show()
