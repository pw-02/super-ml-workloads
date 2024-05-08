import numpy as np
import matplotlib.pyplot as plt

# Number of seconds to run the simulation
total_seconds = 3000

# Define average minibatches processed per second for each condition
base_rate = 10  # Average for Base condition
shade_rate = 15  # Average for Shade condition
super_rate = 20  # Average for SUPER condition

# Simulate the aggregated number of minibatches processed over time
# Using Poisson distribution for each condition
base_minibatches_per_second = np.random.poisson(lam=base_rate, size=total_seconds)
shade_minibatches_per_second = np.random.poisson(lam=shade_rate, size=total_seconds)
super_minibatches_per_second = np.random.poisson(lam=super_rate, size=total_seconds)

# Calculate the cumulative sum of minibatches to get the aggregated number of minibatches processed over time
base_aggregated_minibatches = np.cumsum(base_minibatches_per_second)
shade_aggregated_minibatches = np.cumsum(shade_minibatches_per_second)
super_aggregated_minibatches = np.cumsum(super_minibatches_per_second)

# Create a time array (x-axis) from 0 to total_seconds
time = np.arange(total_seconds)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot Base condition with solid line
plt.plot(time, base_aggregated_minibatches, 'k-', linewidth=2, label='Base')  # 'k-' for black solid line

# Plot Shade condition with dashed line
plt.plot(time, shade_aggregated_minibatches, 'k--', linewidth=2, label='Shade')  # 'k--' for black dashed line

# Plot SUPER condition with dotted line
plt.plot(time, super_aggregated_minibatches, 'k:', linewidth=2, label='SUPER')  # 'k:' for black dotted line

# Label the axes
plt.xlabel('Time (secs)', fontsize=12)
plt.ylabel('Aggregated Number of Minibatches', fontsize=12)

# Set the size of the axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add a title and legend
plt.title('Aggregated Number of Minibatches Processed Over Time', fontsize=12)
plt.legend(fontsize=12)

# Display the plot
plt.show()


print()