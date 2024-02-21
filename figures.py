import matplotlib.pyplot as plt
import numpy as np

# Data
xGPU = [1000, 2000, 3000, 4000, 5000]
batches_per_second = [6644.518272, 13289.03654, 19933.55482, 26578.07309, 33222.59136]
data_load_per_second_kb = [3328.488372, 6656.976744, 9985.465116, 13313.95349, 16642.44186]

# Set up positions for bars
bar_width = 0.35
bar_positions1 = np.arange(len(xGPU))
bar_positions2 = bar_positions1 + bar_width

# Create figure and first y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))
bars1 = ax1.bar(bar_positions1, batches_per_second, bar_width, label='Batches/Second', color='b')

# Create second y-axis
ax2 = ax1.twinx()
bars2 = ax2.bar(bar_positions2, data_load_per_second_kb, bar_width, label='Data Load/Second (KB)', color='r')

# Add labels and title
ax1.set_xlabel('xGPU')
ax1.set_ylabel('Batches/Second', color='b')
ax2.set_ylabel('Data Load/Second (KB)', color='r')
ax1.set_title('GPU Throughput and Data Load Rate')
ax1.set_xticks(bar_positions1 + bar_width / 2)
ax1.set_xticklabels(xGPU)

# Display the figure
plt.show()

print()