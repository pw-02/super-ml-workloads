import matplotlib.pyplot as plt
import pandas as pd

# Define the path to the CSV file
csv_file_path = 'C:\\Users\\pw\\Desktop\\reports\\summary.csv'  # Update this to the correct path

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Extract unique workloads
workloads = df['workload'].unique()

# Define colors for the different types of data
colors = {
    'baseline': '#007E7E',  
    'shade': '#FEA400',
    'litdata': '#FEA400',
    'oracle': '#4C8BB8',   
    'super': '#000000'    
}  

# Create a single figure with subplots in a row
fig, axs = plt.subplots(1, len(workloads), figsize=(4.2 * len(workloads), 3))

for i, workload in enumerate(workloads):
    # Filter DataFrame for the current workload
    workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]

    # Extract labels and capitalize each label
    labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]
    values = workload_df['throughput(batches_per_second)'].tolist()
    
    # Sort labels and values based on the order you want (baseline, shade/litdata, oracle, super)
    order = {'baseline': 1, 'shade': 2, 'litdata': 2, 'super': 3}
    labels, values = zip(*sorted(zip(labels, values), key=lambda x: order.get(x[0].lower(), float('inf'))))
    
    # Plot each bar with the corresponding color in the subplot
    for j, label in enumerate(labels):
        color = colors.get(label.lower(), '#000000')  # Use lowercase to match keys in colors dictionary
        hatch = '/' if label.lower() == 'baseline' else '.' if label.lower() == 'shade' or label.lower() == 'litdata' else '-'  # Custom hatches
        if label.lower() == 'super':
            label = r'$\bf{SUPER}$'  # Set 'super' label to bold
        axs[i].bar(label, values[j], color=color, width=0.7, edgecolor='black', hatch=hatch, alpha=1)
    
    # Add gridlines
    axs[i].grid(axis='y', linestyle='--', alpha=0.7)

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

    # Only add the y-axis label to the first subplot
    if i == 0:
        axs[i].set_ylabel('Aggregated minibatches/second', fontsize=12)

    # Ensure y-axis numbers are shown for all subplots
    axs[i].tick_params(labelleft=True)  # Ensure y-axis numbers are shown

    # Set font size for x-axis labels
    axs[i].tick_params(axis='x', labelsize=12)

# Adjust layout
plt.tight_layout()

# Save the combined figure
plt.savefig('figures/system-comparsion/aggregated_throughput/combined_throughput.png')
plt.show()

print('Combined figure has been created and saved.')
