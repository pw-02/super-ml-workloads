import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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

# Convert the data into the same format as your initial dictionary
result_data = {}

for i, workload in enumerate(workloads):
    # Filter DataFrame for the current workload
    workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]
    
    result_data[workload] = {}
    # Extract labels and capitalize each label
    labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]
    io_values = [val * 100 for val in workload_df['io%'].tolist()]
    transform_values = [val * 100 for val in workload_df['transform%'].tolist()]
    compute_values = [val * 100 for val in workload_df['compute%'].tolist()]
    # io_values = workload_df['io%'].tolist()
    # transform_values = workload_df['transform%'].tolist()
    # compute_values = workload_df['compute%'].tolist()
    
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
# fig, axs = plt.subplots(1, num_workloads, figsize=(4.5 * num_workloads, 3), sharey=False)
fig, axs = plt.subplots(1, len(workloads), figsize=(4.2 * len(workloads), 3))

for i, (workload, data) in enumerate(result_data.items()):
    labels = list(data.keys())
    io_values = [data[label]['IO %'] for label in labels]
    transformation_values = [data[label]['Transformation %'] for label in labels]
    gpu_values = [data[label]['GPU %'] for label in labels]

    # Plotting the bar chart
    ax = axs[i]
   
    # Plotting bars for IO %
    bars_io = ax.barh(labels, io_values, color=io_color, label='IO %', edgecolor='black', alpha=1, hatch='/', linewidth=1.2, height=0.5)
    
    # Plotting bars for Transformation %
    bars_transformation = ax.barh(labels, transformation_values, left=io_values, color=transformation_color, hatch='-', label='Transformation %', edgecolor='black', alpha=1, linewidth=1.2, height=0.5)
    
    # Plotting bars for GPU %
    left_values = [sum(x) for x in zip(io_values, transformation_values)]
    bars_gpu = ax.barh(labels, gpu_values, left=left_values, color=gpu_color, label='GPU %',  hatch='.', edgecolor='black', alpha=1, linewidth=1.2, height=0.5)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add x-axis labels
    # ax.set_xlabel('Percentage of Time (%)', fontsize=12)
    
    # Apply the formatter to the x-axis
    
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    axs[i].tick_params(axis='x', labelsize=12)
    # Adding titles
    if 'resnet18' in workload:
        title = 'ResNet-18/Cifar10'
    elif 'resnet50' in workload:
        title = 'ResNet-50/Cifar10'
    elif '14m' in workload:
        title = 'Pythia-14m/OpenWebText'
    elif '70m' in workload:
        title = 'Pythia-70m/OpenWebText'
    axs[i].set_title(title, fontsize=12)
    # Set y-ticks font size
    ax.tick_params(axis='y', labelsize=12)

# Add legend to the first subplot
# Add legend above all subplots in the center
plt.legend(['I/O%', 'Transform%', 'GPU%'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# Hide the legend from individual subplots
for ax in axs:
     ax.legend().set_visible(False)
# axs[0].legend(['I/O%', 'Transform%', 'GPU%'])

# Adjust layout
plt.tight_layout()

# Save the combined figure
plt.savefig('figures/system-comparsion/time_breakdown/combined_percentage_breakdown.png')
plt.show()

print('Combined figure has been created and saved.')
