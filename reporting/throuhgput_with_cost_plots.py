import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Number of seconds to run the simulation
total_seconds = 3000

# Define average minibatches processed per second for each condition
base_rate = 10
shade_rate = 15
super_rate = 20
extra_case_1_rate = 25

# Simulate the aggregated number of minibatches processed over time
base_minibatches_per_second = np.random.poisson(lam=base_rate, size=total_seconds)
shade_minibatches_per_second = np.random.poisson(lam=shade_rate, size=total_seconds)
super_minibatches_per_second = np.random.poisson(lam=super_rate, size=total_seconds)
extra_case_1_minibatches_per_second = np.random.poisson(lam=extra_case_1_rate, size=total_seconds)

base_aggregated_minibatches = np.cumsum(base_minibatches_per_second)
shade_aggregated_minibatches = np.cumsum(shade_minibatches_per_second)
super_aggregated_minibatches = np.cumsum(super_minibatches_per_second)
extra_case_1_aggregated_minibatches = np.cumsum(extra_case_1_minibatches_per_second)

time = np.arange(total_seconds)

# Function to format the y-axis in thousands
def thousands(x, pos):
    return f'{int(x * 1e-3)}k'

# Function to calculate cost
def calculate_cost(minibatches):
    cost_per_minibatch = 0.05
    return minibatches * cost_per_minibatch

# Calculate costs
base_cost = calculate_cost(base_aggregated_minibatches)
shade_cost = calculate_cost(shade_aggregated_minibatches)
super_cost = calculate_cost(super_aggregated_minibatches)
extra_case_1_cost = calculate_cost(extra_case_1_aggregated_minibatches)

# Define the visual map with line styles, colors, and labels
visual_map = {
    'SUPER(Full Hits)': {'color': '#4C8BB8', 'linestyle': '-', 'label': 'SUPER(Full Hits)',  'alpha': 1.0},
    'Pytorch(Cold Cache)': {'color': '#FF7F0E', 'linestyle': '--', 'label': 'Pytorch(Cold Cache)',  'alpha': 1.0},
    'SUPER(Cold Cache)': {'color': '#007E7E', 'linestyle': '-.', 'label': 'SUPER(Cold Cache)',  'alpha': 1.0},
    'Pytorch(Full Hits)': {'color': 'black', 'linestyle': ':', 'label': 'Pytorch(Full Hits)',  'alpha': 1.0}
}
# # Define the visual map with updated colors, line styles, and labels
# visual_map = {
#     'SUPER(Full Hits)': {'color': '#1b9e77', 'linestyle': '-', 'label': 'SUPER(Full Hits)'},  # Greenish teal
#     'Pytorch(Cold Cache)': {'color': '#d95f02', 'linestyle': '--', 'label': 'Pytorch(Cold Cache)'},  # Deep orange
#     'SUPER(Cold Cache)': {'color': '#7570b3', 'linestyle': '-.', 'label': 'SUPER(Cold Cache)'},  # Muted purple
#     'Pytorch(Full Hits)': {'color': 'black', 'linestyle': ':', 'label': 'Pytorch(Full Hits)'}  # Vibrant pink
# }

# # Define the visual map with black color and different line styles (hatches)
# visual_map = {
#     'SUPER(Full Hits)': {'color': 'black', 'linestyle': '-', 'label': 'SUPER(Full Hits)'},  # Solid line
#     'Pytorch(Cold Cache)': {'color': 'black', 'linestyle': '--', 'label': 'Pytorch(Cold Cache)'},  # Dashed line
#     'SUPER(Cold Cache)': {'color': 'black', 'linestyle': '-.', 'label': 'SUPER(Cold Cache)'},  # Dash-dot line
#     'Pytorch(Full Hits)': {'color': 'black', 'linestyle': ':', 'label': 'Pytorch(Full Hits)'}  # Dotted line
# }


# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5))

# Adjust spacing
plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.3)

# Plot for aggregated minibatches processed
ax1.plot(time, base_aggregated_minibatches, **visual_map['SUPER(Full Hits)'], linewidth=3.5)
ax1.plot(time, extra_case_1_aggregated_minibatches, **visual_map['Pytorch(Cold Cache)'], linewidth=3.5)
ax1.plot(time, super_aggregated_minibatches, **visual_map['SUPER(Cold Cache)'], linewidth=3.5)
ax1.plot(time, shade_aggregated_minibatches, **visual_map['Pytorch(Full Hits)'], linewidth=3.5)

ax1.set_xlabel('Time (secs)', fontsize=11)
ax1.set_ylabel('Aggregated # of Samples Processed', fontsize=11)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
ax1.legend(loc='upper left', fontsize=10)

# Plot for cost over time
ax2.plot(time, base_cost, **visual_map['SUPER(Full Hits)'], linewidth=3)
ax2.plot(time, shade_cost, **visual_map['Pytorch(Cold Cache)'], linewidth=3)
ax2.plot(time, super_cost, **visual_map['SUPER(Cold Cache)'], linewidth=3)
ax2.plot(time, extra_case_1_cost, **visual_map['Pytorch(Full Hits)'], linewidth=3)

ax2.set_xlabel('Time (secs)', fontsize=11)
ax2.set_ylabel('Cost', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax2.yaxis.set_major_formatter(FuncFormatter(thousands))
ax2.legend(loc='upper left', fontsize=11)

plt.tight_layout()
# plt.savefig('my_adjusted_figure_with_visual_map.png', dpi=300, bbox_inches='tight')
plt.show()
