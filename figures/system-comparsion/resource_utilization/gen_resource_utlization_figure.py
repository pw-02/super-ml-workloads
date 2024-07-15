import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

def percent_formatter(x, pos):
    return f'{int(x)}%'

# Define colors, hatches, edgecolors, and alphas for specific subcategories
visual_map = {
    'Baseline': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1},
    'Shade': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
    'Super': {'color': '#000000', 'hatch': '-', 'edgecolor': 'black', 'alpha': 1},
    'Litdata': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
}


# Define the path to the CSV file
csv_file_path = 'C:\\Users\\pw\\Desktop\\reports\\summary.csv'  # Update this to the correct path
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Extract unique workloads
workloads = df['workload'].unique()

# Define the custom order for labels
custom_label_order = ['baseline', 'shade', 'litdata', 'oracle', 'super']

# Convert the data into the same format as your initial dictionary
data = {}

for i, workload in enumerate(workloads):
    # Filter DataFrame for the current workload
    workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]

    data[workload] = {}
    # Extract labels and capitalize each label
    labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]
    cpu_values = workload_df['cpu_usge'].tolist()
    gpu_values = workload_df['gpu_usge'].tolist()

    for idx, value in enumerate(cpu_values):
        cpu_values[idx] = json.loads(value)['mean']
    
    for idx, value in enumerate(gpu_values):
        gpu_values[idx] = json.loads(value)['mean']

    # Construct the dictionary
    for idx, label in enumerate(labels):
        data[workload][label] = {
            'CPU': cpu_values[idx],
            'GPU': gpu_values[idx]
        }
    
categories = list(data.keys())
subcategories = list(set([key for category in data.values() for key in category.keys()]))
metrics = ['CPU', 'GPU']

subcategories.sort(key=lambda x: (custom_label_order.index(x.capitalize()) if x.capitalize() in custom_label_order else float('inf'), x.lower()))

# Create a single figure with subplots in a row
fig, axs = plt.subplots(1, len(workloads), figsize=(4.2 * len(workloads), 3))


for idx, category in enumerate(categories):
    ax = axs[idx]
    x = np.arange(len(metrics))
    bar_width = 0.25  # Width of each bar
    counter = 0
    for i, subcat in enumerate(subcategories):
        if subcat in data[category]:
            y = [data[category][subcat][metric] for metric in metrics]
            visual_attr = visual_map.get(subcat, {})  # Get visual attributes from the map, or empty dictionary if not defined
            ax.bar(x + counter * bar_width, y, bar_width, label=subcat, linewidth=1.2,
                color=visual_attr.get('color', '#4C8BB8'),  # Default color if not specified
                hatch=visual_attr.get('hatch', None),  # No hatch if not specified
                edgecolor=visual_attr.get('edgecolor', 'black'),  # Black edgecolor if not specified
                alpha=visual_attr.get('alpha', 1.0))  # Full alpha if not specified
            counter +=1
        else:
            continue
    
    # Replace x-axis ticks with workload names
    ax.set_xticklabels(metrics)
    ax.set_xticks(x + bar_width / 2)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if idx == 0:
        ax.set_ylabel('Resource Utilization (%)', fontsize=12)
    
    ax.tick_params(axis='y', labelsize=12)
    
    # Adding titles
    if 'resnet18' in category:
        title = 'ResNet-18/Cifar10'
    elif 'resnet50' in category:
        title = 'ResNet-50/Cifar10'
    elif '14m' in category:
        title = 'Pythia-14m/OpenWebText'
    elif '70m' in category:
        title = 'Pythia-70m/OpenWebText'

    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylim([0, 100])
    
    legend_labels = []
    for i, subcat in enumerate(subcategories):
        if subcat in data[category]:
            if 'Super' in subcat and subcat in data[category]:
                legend_labels.append(r'$\bf{SUPER}$')
                # subcat[i] = r'$\bf{SUPER}$'
            else:
                legend_labels.append(subcat)
    ax.legend(legend_labels,fontsize=10)

plt.tight_layout()
plt.savefig('figures/system-comparsion/resource_utilization/resource_utilization.png', bbox_inches='tight')
plt.show()

print('Combined figure has been created and saved.')
