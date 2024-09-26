# Define the visual map with properties for each data loader
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

visual_map = {
    r'$\bf{SUPER}$': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'CoorDL': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'Shade': {'color': '#4C8BB8', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'LiData': {'color': '#FF7F0E', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1.0},
}

# Define the workloads
figure_data: Dict[str, Dict[str, float]] = {}
figure_data['ResNet-18/Cifar10'] = {'CoorDL': 1989,'Shade': 1989, r'$\bf{SUPER}$': 4179}
figure_data['ResNet-50/Cifar10'] = {'CoorDL': 202,'Shade': 202, r'$\bf{SUPER}$': 489}
figure_data['Albef/COCO'] ={'CoorDL': 202,'Shade': 202, r'$\bf{SUPER}$': 489}
figure_data['Pythia-14m/OpenWebText'] = {'LiData': 574, r'$\bf{SUPER}$': 1117}

# Plot the results
fig, axs = plt.subplots(1, 4, figsize=(16, 2.3))  # Create a 1x4 grid of subplots wXh=1x4
axs = axs.flatten()

bar_width = 0.5
for idx, workload in enumerate(list(figure_data.keys())):
    ax = axs[idx]
    
    # Extract the relevant data loaders for the current workload
    loaders = list(figure_data[workload].keys())  # Get only the data loaders present for this workload
    data_to_plot = [figure_data[workload][dl] for dl in loaders]  # Get the corresponding values
    
    # Create the bar plot for this workload
    bars = ax.bar(loaders, data_to_plot, color=[visual_map[dl]['color'] for dl in loaders], 
                  edgecolor=[visual_map[dl]['edgecolor'] for dl in loaders], 
                  hatch=[visual_map[dl]['hatch'] for dl in loaders], 
                  width=bar_width, alpha=1.0)
    
    # Set titles and labels
    ax.set_title(f'{workload}', fontsize=10)
    if idx == 0:
        ax.set_ylabel('Throughput (samples/s)', fontsize=10)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()