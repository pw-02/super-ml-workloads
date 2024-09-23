import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Function to format the y-axis as percentages
def percent_formatter(x, pos):
    return f'{int(x)}%'

# Define the visual map
visual_map = {
    'io': {'color': '#007E7E', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0},
    'transform': {'color': '#FEA400', 'hatch': '', 'edgecolor': 'black', 'alpha': 1.0},
    'gpu': {'color': '#4C8BB8', 'hatch': '..', 'edgecolor': 'black', 'alpha': 1.0},
}

# Data for the specified workloads
result_data = {
    'ResNet18': {
        'CoorDL': {'IO %': 75, 'Transformation %': 15, 'GPU %': 10},
        'SHADE': {'IO %': 80, 'Transformation %': 10, 'GPU %': 10},
        'SUPER': {'IO %': 50, 'Transformation %': 20, 'GPU %': 30},
    },
    'ResNet50': {
        'CoorDL': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
        'SHADE': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
        'SUPER': {'IO %': 40, 'Transformation %': 20, 'GPU %': 40},
    },
    'Pythia14': {
        'LitData': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
        'SUPER': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
    },
    'Pythia60': {
        'LitData': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
        'SUPER': {'IO %': 70, 'Transformation %': 10, 'GPU %': 20},
    }
}

# Set up the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(6.5, 3.8))
axs = axs.flatten()  # Flatten the array for easy indexing
bar_width = 0.5  # Width of the bars
# Iterate over the models and create a bar chart for each
for i, (model_name, workloads) in enumerate(result_data.items()):
    labels = list(workloads.keys())
    io_values = [workloads[label]['IO %'] for label in labels]
    transform_values = [workloads[label]['Transformation %'] for label in labels]
    gpu_values = [workloads[label]['GPU %'] for label in labels]

    # Create the stacked bar chart
    axs[i].bar(labels, io_values, label='IO %', color=visual_map['io']['color'], hatch=visual_map['io']['hatch'], width=bar_width)
    axs[i].bar(labels, transform_values, bottom=io_values, label='Transformation %', color=visual_map['transform']['color'], hatch=visual_map['transform']['hatch'],width=bar_width)
    axs[i].bar(labels, gpu_values, bottom=np.array(io_values) + np.array(transform_values), label='GPU %', color=visual_map['gpu']['color'], hatch=visual_map['gpu']['hatch'],width=bar_width)

    # Customize each chart
    axs[i].set_ylabel('Time Breakdown (%)')
    axs[i].yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    axs[i].set_ylim(0, 100)
    axs[i].set_title(f'{model_name}', fontsize=10)

# Create a single legend for the entire figure above the subplots
handles, labels = axs[0].get_legend_handles_labels()  # Get handles and labels from the first subplot
fig.legend(handles, labels, loc='upper center', ncol=3)


# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the legend
plt.subplots_adjust(left=0.129, right=0.938, top=0.817, bottom=0.102, wspace=0.364, hspace=0.443)

plt.show()
