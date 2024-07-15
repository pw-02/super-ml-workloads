import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

def percent_formatter(x, pos):
    return f'{int(x)}%'

io_color ='#007E7E'  
transformation_color = '#000000' #4C8BB8
gpu_color = '#FEA400'

# Define the path to the CSV file
csv_file_path = 'C:\\Users\\pw\\Desktop\\reports\\summary.csv'  # Update this to the correct path
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Extract unique workloads
workloads = df['workload'].unique()

# Define the custom order for labels
custom_label_order = ['baseline', 'shade', 'litdata', 'oracle', 'super']

# Convert the data into the same format as your initial dictionary
result_data = {}

for i, workload in enumerate(workloads):
    # Filter DataFrame for the current workload
    workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]
    # workload_df = df[(df['workload'] == workload)]

    result_data[workload] = {}
    # Extract labels and capitalize each label
    labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]
    cpu_values = workload_df['cpu_usge'].tolist()
    gpu_values = workload_df['gpu_usge']

    for idx, value in enumerate(cpu_values):
        cpu_values[idx] = json.loads(value)['mean']
    
    for idx, value in enumerate(gpu_values):
        gpu_values[idx] = json.loads(value)['mean']
    
    # Construct the dictionary
    for idx, label in enumerate(labels):
        result_data[workload][label] = {
            'CPU %': cpu_values[idx],
            'GPU %': gpu_values[idx]
        }
        
# Number of workloads
num_workloads = len(result_data)

# Create a single figure with subplots in a row
fig, axs = plt.subplots(1, len(workloads), figsize=(4.2 * len(workloads), 3))

# Lists to store handles and labels for the legend
all_handles = []
all_labels = []

for i, (workload, data) in enumerate(result_data.items()):
    ax = axs[i]  # Define ax here

    labels = list(data.keys())

    io_values = [data[label]['CPU %'] for label in labels]
    gpu_values = [data[label]['GPU %'] for label in labels]

    # Sort the labels based on the custom order
    labels_sorted = sorted(labels, key=lambda x: custom_label_order.index(x.lower()))

    # Reorder the data accordingly
    io_values_sorted = [io_values[labels.index(label)] for label in labels_sorted]
    gpu_values_sorted = [gpu_values[labels.index(label)] for label in labels_sorted]
    
    for idx, label in enumerate(labels_sorted):
        if label.lower() == 'super':
            labels_sorted[idx] = r'$\bf{SUPER}$'
            
    # Width of each bar
    bar_width = 0.35

    # X positions for the bars
    x = np.arange(len(labels_sorted))

    # Plotting bars for IO %
    # Plotting bars for IO %
    bars_io = ax.bar(x - bar_width/2, io_values_sorted, width=bar_width, color=io_color, label='CPU', edgecolor='black', alpha=1, hatch='/', linewidth=1.2)

    # Plotting bars for GPU %
    bars_gpu = ax.bar(x + bar_width/2, gpu_values_sorted, width=bar_width, color=gpu_color, label='GPU', hatch='.', edgecolor='black', alpha=1, linewidth=1.2)

    # Replace x-axis ticks with workload names
    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)


    # Apply the formatter to the y-axis
    if i == 0:
        ax.set_ylabel('Resource Utilization (%)', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
        
    # Adding titles
    if 'resnet18' in workload:
        title = 'ResNet-18/Cifar10'
    elif 'resnet50' in workload:
        title = 'ResNet-50/Cifar10'
    elif '14m' in workload:
        title = 'Pythia-14m/OpenWebText'
    elif '70m' in workload:
        title = 'Pythia-70m/OpenWebText'

    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    
    # Get handles and labels for legend
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)
    

    
    # Set y-axis limits to 0 and 100
    ax.set_ylim([0, 100])
    ax.legend(all_handles, all_labels, fontsize=10)

plt.tight_layout()


plt.savefig('figures/system-comparsion/resource_utilization/combined_resource_utilization.png', bbox_inches='tight')
plt.show()

print('Combined figure has been created and saved.')
