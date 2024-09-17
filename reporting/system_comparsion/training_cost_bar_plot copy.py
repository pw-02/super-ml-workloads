import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate dummy data
np.random.seed(0)

# Define the visual map with properties for each data loader
visual_map = {
    'SUPER(Full Hits)': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'SUPER(Cold Cache)': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'Pytorch(Full Hits)': {'color': '#4C8BB8', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
    'Pytorch(Cold Cache)': {'color': '#FF7F0E', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1.0},
}

# visual_map = {
#     'SUPER(Full Hits)': {'color': '#2A3E5C', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},  # Muted navy blue
#     'SUPER(Cold Cache)': {'color': '#4C72B0', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},  # Lighter blue with dots
#     'Pytorch(Full Hits)': {'color': '#C44E52', 'hatch': '--', 'edgecolor': 'black', 'alpha': 1.0},  # Muted red with dashes
#     'Pytorch(Cold Cache)': {'color': '#DD8452', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},  # Soft orange with crosshatch
# }

# visual_map = {
#     'SUPER(Full Hits)': {'color': '#1f77b4', 'hatch': 'xx', 'edgecolor': 'black', 'alpha': 1.0},  # Soft blue with crosshatch
#     'SUPER(Cold Cache)': {'color': '#aec7e8', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},  # Lighter blue with diagonal lines
#     'Pytorch(Full Hits)': {'color': '#ff9896', 'hatch': '++', 'edgecolor': 'black', 'alpha': 1.0},  # Soft pink with plus signs
#     'Pytorch(Cold Cache)': {'color': '#d62728', 'hatch': '||', 'edgecolor': 'black', 'alpha': 1.0},  # Deep red with vertical lines
# }







data_loaders = list(visual_map.keys())
workloads = ['ResNet-18/Cifar10', 'ResNet-50/Cifar10', 'Pythia-14m/OpenWebText', 'Pythia-70m/OpenWebText']
data = {
    'workload': np.random.choice(workloads, size=400),
    'datalaoder': np.random.choice(data_loaders, size=400),
    'throughput(batches_per_second)': np.random.normal(loc=50, scale=10, size=400)
}

df = pd.DataFrame(data)

# Aggregate data: mean throughput per workload and data loader
agg_df = df.groupby(['workload', 'datalaoder'])['throughput(batches_per_second)'].mean().unstack().fillna(0)

# Create a single figure with subplots
fig, ax = plt.subplots(figsize=(8, 3.75))

# Define plot parameters
num_data_loaders = len(data_loaders)
bar_width = 0.2
x = np.arange(len(workloads))  # the x locations for the groups
colors = [visual_map[loader]['color'] for loader in data_loaders]
hatches = [visual_map[loader]['hatch'] for loader in data_loaders]

# Plot each data loader as a group of bars
for i, (data_loader, color, hatch) in enumerate(zip(data_loaders, colors, hatches)):
    # Get throughput data for each data loader
    throughput_data = agg_df[data_loader].reindex(workloads, fill_value=0)
    ax.bar(x + i * bar_width, throughput_data, bar_width,
           color=color, edgecolor='black', hatch=hatch, label=data_loader, alpha=0.95)

# Add labels, title, and legend
ax.set_ylabel('Aggregated Throughput (samples/sec)', fontsize=12)
ax.set_xticks(x + bar_width * (num_data_loaders - 1) / 2)
ax.set_xticklabels(workloads, rotation=-0, ha='center', fontsize=10)
ax.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15))

# Adjust layout to prevent overlap
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the combined figure
plt.show()

print('Figure with combined throughput data has been created and saved.')
