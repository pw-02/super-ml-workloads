import matplotlib.pyplot as plt

# Data
data = {
    'ImageNet-1k': {
        'EC Redis': 2733.55852080,
        'EC Serverless': 11159.87500000,
        'InfiniStore': 163.12173482,
    }
}

visual_map = {
    'EC Redis': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
    'EC Serverless': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1},
    'InfiniStore': {'color': '#4C8BB8', 'hatch': '-', 'edgecolor': 'black', 'alpha': 1},
}

for dataset_name in data:

    services = list(data[dataset_name].keys())
    # Extracting the keys and values for plotting
    services = list(data[dataset_name].keys())
    costs = list(data[dataset_name].values())

    # Set the figure size and style
    plt.figure(figsize=(4.5, 2.2))

    # Create the bar plot with custom styles
    bars = []
    for i, service in enumerate(services):
        style = visual_map[service]
        bar = plt.bar(service, costs[i], color=style['color'], edgecolor=style['edgecolor'],
                    hatch=style['hatch'], alpha=style['alpha'], label=service)
        bars.append(bar)

    # Add title and labels
    plt.ylabel('Cost ($)', fontsize=12)

    # Remove x-axis tick labels
    plt.xticks([])

    # Add legend
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.yticks(fontsize=12)

    # plt.legend(loc='upper right')
    plt.ylim([0, 12500])
    # Add labels on top of the bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar[0].get_height()
        fontweight = 'bold' if i == len(bars) - 1 else 'normal'
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, height, f'${cost:.2f}', 
                ha='center', va='bottom', fontsize=12, weight=fontweight)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f'figures/cost_analysis/{dataset_name}_1month_storage_cost.png', bbox_inches='tight')
    plt.show()
