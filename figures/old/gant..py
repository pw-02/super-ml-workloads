import matplotlib.pyplot as plt
import numpy as np

# Data (example values)
num_mini_batches = 15
start_times = np.random.randint(0, 100, num_mini_batches)  # Random start times for each mini-batch
io_time = np.random.randint(1, 10, num_mini_batches)  # Random data for I/O time
transform_time = np.random.randint(1, 10, num_mini_batches)  # Random data for transformation time
gpu_time = np.random.randint(1, 10, num_mini_batches)  # Random data for GPU time

# Calculate end times for each mini-batch
end_times = start_times + io_time + transform_time + gpu_time

# Plotting the Gantt chart
plt.figure(figsize=(4, 3))

# Plot I/O time
plt.barh(np.arange(1, num_mini_batches + 1), io_time, left=start_times, color='#007E7E', label='I/O Time')

# Plot transformation time
plt.barh(np.arange(1, num_mini_batches + 1), transform_time, left=start_times + io_time, color='#000000', label='Transform Time')

# Plot GPU time
plt.barh(np.arange(1, num_mini_batches + 1), gpu_time, left=start_times + io_time + transform_time, color='#FEA400', label='GPU Time')

plt.subplots_adjust(left=0.13, right=0.967, top=0.979, bottom=0.16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding labels and title
plt.xlabel('Time', fontsize=10)
plt.ylabel('Mini-batch Index')
plt.legend()

# Show plot
plt.savefig('gant.png')