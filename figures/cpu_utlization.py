import matplotlib.pyplot as plt
import numpy as np

# Sample data
# Time values (0 to 3000 seconds)
time = np.linspace(0, 3000, num=100)  # Create an array of 100 points from 0 to 3000 seconds

# Sample CPU utilization data for PyTorch, Shade, and Super over time
# You can replace these with actual data arrays
cpu_pytorch = np.random.uniform(20, 80, size=len(time))  # Sample random data, replace with actual data
cpu_shade = np.random.uniform(20, 80, size=len(time))    # Sample random data, replace with actual data
cpu_super = np.random.uniform(20, 80, size=len(time))    # Sample random data, replace with actual data

# Create a figure and axis
plt.figure()

# Plot the data for each system
plt.plot(time, cpu_pytorch, label='PyTorch', marker='o')
plt.plot(time, cpu_shade, label='Shade', marker='s')
plt.plot(time, cpu_super, label='Super', marker='^')

# Add labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('CPU Utilization (%)')
plt.title('CPU Utilization Over Time (3000 seconds)')

# Add legend
plt.legend()

# Display the plot
plt.show()


print()