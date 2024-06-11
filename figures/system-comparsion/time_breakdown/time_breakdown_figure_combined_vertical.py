import matplotlib.pyplot as plt
import pandas as pd

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
    io_values = [val * 100 for val in workload_df['io%'].tolist()]
    transform_values = [val * 100 for val in workload_df['transform%'].tolist()]
    compute_values = [val * 100 for val in workload_df['compute%'].tolist()]
    
    # Construct the dictionary
    for idx, label in enumerate(labels):
        result_data[workload][label] = {
            'IO %': io_values[idx],
            'Transformation %': transform_values[idx],
            'GPU %': compute_values[idx]
        }

# Number of workloads
num_workloads = len(result_data)

# Create a single figure with subplots in a row
fig, axs = plt.subplots(1, len(workloads), figsize=(4.2 * len(workloads), 3))

# Lists to store handles and labels for the legend
all_handles = []
all_labels = []

for i, (workload, data) in enumerate(result_data.items()):
    labels = list(data.keys())

    io_values = [data[label]['IO %'] for label in labels]
    transformation_values = [data[label]['Transformation %'] for label in labels]
    gpu_values = [data[label]['GPU %'] for label in labels]

    # Sort the labels based on the custom order
    labels_sorted = sorted(labels, key=lambda x: custom_label_order.index(x.lower()))

    # Reorder the data accordingly
    io_values_sorted = [io_values[labels.index(label)] for label in labels_sorted]
    transformation_values_sorted = [transformation_values[labels.index(label)] for label in labels_sorted]
    gpu_values_sorted = [gpu_values[labels.index(label)] for label in labels_sorted]
    
    for idx, label in enumerate(labels_sorted):
        if label.lower() == 'super':
            labels_sorted[idx] = r'$\bf{SUPER}$'
            
    # Plotting the bar chart
    ax = axs[i]

    # Plotting bars for IO %
    bars_io = ax.bar(labels_sorted, io_values_sorted, color=io_color, label='IO %', edgecolor='black', alpha=1, hatch='/', linewidth=1.2)

    # Plotting bars for Transformation %
    bars_transformation = ax.bar(labels_sorted, transformation_values_sorted, bottom=io_values_sorted, color=transformation_color, hatch='-', label='Transformation %', edgecolor='black', alpha=1, linewidth=1.2)

    # Plotting bars for GPU %
    bottom_values = [0] * len(labels_sorted)
    bars_gpu = ax.bar(labels_sorted, gpu_values_sorted, bottom=[sum(x) for x in zip(io_values_sorted, transformation_values_sorted)], color=gpu_color, label='GPU %', hatch='.', edgecolor='black', alpha=1, linewidth=1.2)

    # Add gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Apply the formatter to the y-axis
    if i == 0:
        ax.set_ylabel('Percentage of Time (%)', fontsize=12)
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

# Create a legend subplot above all others
legend_ax = fig.add_subplot(111, frame_on=False)
legend_ax.axis('off')
legend_ax.legend(all_handles, all_labels, loc='upper left', bbox_to_anchor=(0.001, 1.5), ncol=3, fontsize=12)

# Adjust the position of the legend subplot
fig.subplots_adjust(bottom=0.4)

plt.tight_layout()

plt.savefig('figures/system-comparsion/time_breakdown/combined_percentage_breakdown.png', bbox_inches='tight')
plt.show()

print('Combined figure has been created and saved.')
