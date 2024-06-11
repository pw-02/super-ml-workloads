import matplotlib.pyplot as plt
import os
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

# Loop through each unique workload to create figures
for workload in workloads:
    # Filter DataFrame for the current workload
    #workload_df = df[df['workload'] == workload]
    workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]

    # Extract labels and capitalize each label
    labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]
    values = workload_df['throughput(batches_per_second)'].tolist()
    
    # Sort labels and values based on the order you want (baseline, shade/litdata, oracle, super)
    # order = {'baseline': 1, 'shade': 2, 'litdata': 2, 'oracle': 3, 'super': 4}
    order = {'baseline': 1, 'shade': 2, 'litdata': 2, 'super': 3}

    labels, values = zip(*sorted(zip(labels, values), key=lambda x: order.get(x[0].lower(), float('inf'))))
    
 
    # Create figure
    plt.figure(figsize=(4.2, 3))
    
    # Plot each bar with the corresponding color
    for i, label in enumerate(labels):
        color = colors.get(label.lower(), '#000000')  # Use lowercase to match keys in colors dictionary
        hatch = '/' if label.lower() == 'baseline' else '.' if label.lower() == 'shade' or label.lower() == 'litdata' else '-'  # Custom hatches
        if label.lower() == 'super':
            label = r'$\bf{SUPER}$'  # Set 'super' label to bold
        plt.bar(label, values[i], color=color, width=0.7, edgecolor='black', hatch=hatch, alpha=1)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding labels and title
    plt.ylabel('Aggregated minibatches/second', fontsize=12)

    # Set font size for bar labels
    plt.xticks(fontsize=12)

    # Add legend
    # plt.legend(labels)

    # Save the figure
    plt.savefig(f'figures\\system-comparsion\\aggregated_throughput\\{workload}_throughput.png')
    plt.close()

print('Figures have been created and saved.')