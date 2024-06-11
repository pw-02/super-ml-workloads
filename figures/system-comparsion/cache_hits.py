import numpy as np
import matplotlib.pyplot as plt

# Number of seconds to run the simulation
total_seconds = 300

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

pytorch_color ='#E5E5E7'  
shade_color = '#B2C6B6'   
super_color = '#0B3041'    


# Create the plot
plt.figure(figsize=(4.5, 4))

# Plot Base condition with solid line
plt.plot(time, base_aggregated_minibatches, 'k-', linewidth=2, label='Base', color=pytorch_color)  # 'k-' for black solid line

# Plot Shade condition with dashed line
plt.plot(time, shade_aggregated_minibatches, 'k--', linewidth=2, label='Shade', color=shade_color)  # 'k--' for black dashed line

# Plot SUPER condition with dotted line
plt.plot(time, super_aggregated_minibatches, 'k:', linewidth=2, label='SUPER', color=super_color)  # 'k:' for black dotted line

# Label the axes
plt.xlabel('Time (secs)', fontsize=12)
plt.ylabel('Aggregated Cache Hits', fontsize=12)

# Set the size of the axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(left=0.18, right=0.9, top=0.9, bottom=0.126)

# # Add a title and legend
# plt.legend(fontsize=12)

# Display the plot
# plt.show()

plt.savefig('figures/cachehits/resnet18_cachehits.png')

