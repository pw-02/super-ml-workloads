import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Number of seconds to run the simulation
total_seconds = 3000

# Define average minibatches processed per second for each condition
coor_rate = 10
shade_rate = 15
super_rate = 20

# Simulate the aggregated number of minibatches processed over time
super_minibatches_per_second = np.random.poisson(lam=super_rate, size=total_seconds)
shade_minibatches_per_second = np.random.poisson(lam=shade_rate, size=total_seconds)
coor_minibatches_per_second = np.random.poisson(lam=coor_rate, size=total_seconds)

shade_aggregated_minibatches = np.cumsum(shade_minibatches_per_second)
super_aggregated_minibatches = np.cumsum(super_minibatches_per_second)
coordl_aggregated_minibatches = np.cumsum(coor_minibatches_per_second)

time = np.arange(total_seconds)

# Function to format the y-axis in thousands
def thousands(x, pos):
    return f'{int(x * 1e-3)}k'

# Function to calculate cost
def calculate_cost(minibatches):
    cost_per_minibatch = 0.05
    return minibatches * cost_per_minibatch

# Calculate costs
shade_cost = calculate_cost(shade_aggregated_minibatches)
super_cost = calculate_cost(super_aggregated_minibatches)
coordl_cost = calculate_cost(coordl_aggregated_minibatches)
line_width =1.5
# Define the visual map with line styles, colors, and labels
visual_map = {
     r'$\bf{SUPER}$': {'color': '#4C8BB8', 'linestyle': '-', 'label':  r'$\bf{SUPER}$',  'alpha': 1.0},
    'Shade': {'color': '#FF7F0E', 'linestyle': '--', 'label': 'Shade',  'alpha': 1.0},
    'CoorDL': {'color': '#007E7E', 'linestyle': '-.', 'label': 'CoorDL',  'alpha': 1.0},
}
# Define the visual map with line styles, colors, and labels
visual_map = {
     r'$\bf{SUPER}$': {'color': 'black', 'linestyle': '-', 'label':  r'$\bf{SUPER}$',  'alpha': 1.0, 'linewidth': line_width},
    'Shade': {'color': 'black', 'linestyle': '--', 'label': 'Shade',  'alpha': 1.0,'linewidth': line_width},
    'CoorDL': {'color': 'black', 'linestyle': ':', 'label': 'CoorDL',  'alpha': 1.0,'linewidth': line_width},
}
# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.1, 3.2))

# Adjust spacing
plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.3)

# Plot for aggregated minibatches processed
ax1.plot(time, shade_aggregated_minibatches, **visual_map['Shade'])
ax1.plot(time, coordl_aggregated_minibatches, **visual_map['CoorDL'])
ax1.plot(time, super_aggregated_minibatches, **visual_map[r'$\bf{SUPER}$'])

ax1.set_xlabel('Time (secs)', fontsize=11)
ax1.set_ylabel('Aggregated Samples Processed', fontsize=11)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
ax1.legend(loc='upper left', fontsize=10)

# Plot for cost over time
ax2.plot(time, shade_cost, **visual_map['Shade'])
ax2.plot(time, coordl_cost, **visual_map['CoorDL'])
ax2.plot(time, super_cost, **visual_map[r'$\bf{SUPER}$'])

ax2.set_xlabel('Time (secs)', fontsize=11)
ax2.set_ylabel('Cost', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.yaxis.set_major_formatter(FuncFormatter(thousands))
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
# plt.savefig('my_adjusted_figure_with_visual_map.png', dpi=300, bbox_inches='tight')
plt.show()
