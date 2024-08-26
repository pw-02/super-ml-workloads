import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Number of seconds to run the simulation
total_seconds = 3000

# Define average minibatches processed per second for each condition
base_rate = 10  # Average for Base condition
shade_rate = 15  # Average for Shade condition
super_rate = 20  # Average for SUPER condition
new_system1_rate = 12  # Average for NewSystem1 condition
new_system2_rate = 18  # Average for NewSystem2 condition

# Simulate the aggregated number of minibatches processed over time
# Using Poisson distribution for each condition
base_minibatches_per_second = np.random.poisson(lam=base_rate, size=total_seconds)
shade_minibatches_per_second = np.random.poisson(lam=shade_rate, size=total_seconds)
super_minibatches_per_second = np.random.poisson(lam=super_rate, size=total_seconds)
new_system1_minibatches_per_second = np.random.poisson(lam=new_system1_rate, size=total_seconds)
new_system2_minibatches_per_second = np.random.poisson(lam=new_system2_rate, size=total_seconds)

# Calculate the cumulative sum of minibatches to get the aggregated number of minibatches processed over time
base_aggregated_minibatches = np.cumsum(base_minibatches_per_second)
shade_aggregated_minibatches = np.cumsum(shade_minibatches_per_second)
super_aggregated_minibatches = np.cumsum(super_minibatches_per_second)
new_system1_aggregated_minibatches = np.cumsum(new_system1_minibatches_per_second)
new_system2_aggregated_minibatches = np.cumsum(new_system2_minibatches_per_second)

# Create a time array (x-axis) from 0 to total_seconds
time = np.arange(total_seconds)

# Define a formatter function for the y-axis
def thousands(x, pos):
    """The two args are the value and tick position."""
    return f'{int(x * 1e-3)}k'

# Create the plot
plt.figure(figsize=(6,4))

# Plot Base condition with solid black line
plt.plot(time, base_aggregated_minibatches, 'k-', linewidth=2, label='Base')  # 'k-' for black solid line

# Plot Shade condition with dashed blue line
plt.plot(time, shade_aggregated_minibatches, 'b--', linewidth=2, label='Shade')  # 'b--' for blue dashed line

# Plot SUPER condition with dotted green line
plt.plot(time, super_aggregated_minibatches, 'g:', linewidth=2, label='SUPER')  # 'g:' for green dotted line

# Plot NewSystem1 condition with dash-dot orange line
plt.plot(time, new_system1_aggregated_minibatches, 'orange', linestyle='-.', linewidth=2, label='Full Hits (Redis)')  # Orange dash-dot line

# Plot NewSystem2 condition with solid red line
plt.plot(time, new_system2_aggregated_minibatches, 'r-', linewidth=2, label='Cache Hits (Serverless)')  # Red solid line

# Label the axes
plt.xlabel('Time (secs)', fontsize=12)
plt.ylabel('Aggregated Number of Samples Processed', fontsize=11)

# Set the size of the axis values
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Apply the custom formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands))

# Add a title and legend
plt.legend(fontsize=10)

# Display the plot
plt.show()
