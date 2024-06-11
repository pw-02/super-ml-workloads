import matplotlib.pyplot as plt

baseline_color ='#007E7E'  
sota_color = '#FEA400'
oracle_color = '#4C8BB8'   
super_color = '#000000' 

result_data = {
    'eval_resnet18_cifar10': {'Baseline': 72.5122, 'SHADE': 65, 'Oracle': 31, r'$\bf{SUPER}$': 48},
    'eval_resnet50_imagenet': {'Baseline': 64.5, 'SHADE': 69, 'Oracle': 39.1202, r'$\bf{SUPER}$': 55},
    'eval_owt_pythia14m': {'Baseline': 26,'LitData': 69,'Oracle': 15,r'$\bf{SUPER}$': 16.7},
    'eval_owt_pythia70m':  {'Baseline': 22, 'LitData': 69, 'Oracle': 15.7, r'$\bf{SUPER}$': 16.5} 
    }


for workload, data in result_data.items():

    # Extract labels and values
    labels = list(data.keys())
    values = list(data.values())

    # Plotting the bar chart
    #plt.figure(figsize=(4.5, 4))
    plt.figure(figsize=(4.5, 3))
    bar_width = 0.7
    #Plotting the bar graph with customized colors and patterns
    plt.bar(labels[0], values[0], color=baseline_color, width=bar_width, edgecolor='black', label='PyTorch', hatch='/', alpha=1)
    plt.bar(labels[1], values[1], color=sota_color, width=bar_width, edgecolor='black', label='SHADE', hatch='.', alpha=1)
    plt.bar(labels[2], values[2], color=oracle_color, width=bar_width, edgecolor='black', label='SUPER', hatch='-', alpha=1)
    plt.bar(labels[3], values[3], color=super_color, width=bar_width, edgecolor='black', label='SUPER', hatch='-', alpha=1)

    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Reduce white space between bars
    plt.subplots_adjust(left=0.125, right=0.9, top=0.88, bottom=0.17)

    # Adding labels and title
    plt.ylabel('CPU Utilization (%)', fontsize=12)

    # Set font size for bar labels
    plt.xticks(fontsize=12, weight='normal')

    # Adding legend
    plt.legend(labels)

    # Save the plot as a PNG file
    plt.savefig(f'figures/system-comparsion/{workload}/cpu_utilization.png')