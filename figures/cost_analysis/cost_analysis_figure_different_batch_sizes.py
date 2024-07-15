import matplotlib.pyplot as plt
import numpy as np
from storage_costs_calculations import system_comaprsion

# Assuming data is fetched using your main function
data = system_comaprsion()


# Determine unique keys for plotting
keys = list(data[0].keys())
keys.remove('percentage')  # Remove 'percentage' from keys for bar plotting

# Bar colors and styles
visual_map = {
    'aws_redis_cost': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_8_batch_size': {'color': '#4C8BB8', 'hatch': '-', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_64_batch_size': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_128_batch_size': {'color': '#FF5733', 'hatch': '+', 'edgecolor': 'black', 'alpha': 0.8},
}

# Extract data for plotting
percentages = [entry['percentage'] for entry in data]
bar_data = {key: [entry[key] for entry in data] for key in keys}

# Bar width and positions
num_keys = len(keys)
bar_width = 0.2
r_positions = [np.arange(len(percentages)) + i * bar_width for i in range(num_keys)]

# Plotting
plt.figure(figsize=(10, 6))

for i, key in enumerate(keys):
    plt.bar(r_positions[i], bar_data[key], color=visual_map[key]['color'], width=bar_width, edgecolor=visual_map[key]['edgecolor'], hatch=visual_map[key]['hatch'], label=key.replace('_', ' ').title(), alpha=visual_map[key]['alpha'])
    # # Adding values above the bars
    # for r, cost in zip(r_positions[i], bar_data[key]):
    #     plt.text(r, cost + 20, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)

# Adding titles and labels
plt.xlabel('% Dataset Cached', fontsize=12)
plt.ylabel('Cost ($)', fontsize=12)
plt.xticks([r + bar_width * (num_keys - 1) / 2 for r in range(len(percentages))], percentages, fontsize=10)
plt.legend()
plt.title('Comparison of Costs')
plt.tight_layout()
plt.savefig('figures/cost_analysis.png', bbox_inches='tight')
plt.show()
