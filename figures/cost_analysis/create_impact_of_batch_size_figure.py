import matplotlib.pyplot as plt
from storage_costs_calculations import impact_of_differnt_batch_sizes
import numpy as np

data = impact_of_differnt_batch_sizes()

# Bar colors and styles
visual_map = {
    'aws_redis_cost': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_8_batch_size': {'color': '#4C8BB8', 'hatch': '-', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_128_batch_size': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 0.8},
    'severless_cost_256_batch_size': {'color': '#FF5733', 'hatch': '+', 'edgecolor': 'black', 'alpha': 0.8},
}

# Extract data for plotting
percentages = [entry['percentage'] for entry in data]
aws_redis_cost = [entry['aws_redis_cost'] for entry in data]
severless_cost_8 = [entry['severless_cost_8_batch_size'] for entry in data]
severless_cost_128 = [entry['severless_cost_128_batch_size'] for entry in data]
severless_cost_256 = [entry['severless_cost_256_batch_size'] for entry in data]

# Bar width and positions
bar_width = 0.2
r1 = np.arange(len(percentages))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting
plt.figure(figsize=(10, 6))

plt.bar(r1, aws_redis_cost, color=visual_map['aws_redis_cost']['color'], width=bar_width, edgecolor=visual_map['aws_redis_cost']['edgecolor'], hatch=visual_map['aws_redis_cost']['hatch'], label='AWS Redis Cost', alpha=visual_map['aws_redis_cost']['alpha'])
plt.bar(r2, severless_cost_8, color=visual_map['severless_cost_8_batch_size']['color'], width=bar_width, edgecolor=visual_map['severless_cost_8_batch_size']['edgecolor'], hatch=visual_map['severless_cost_8_batch_size']['hatch'], label='Serverless Cost (8 batch size)', alpha=visual_map['severless_cost_8_batch_size']['alpha'])
plt.bar(r3, severless_cost_128, color=visual_map['severless_cost_128_batch_size']['color'], width=bar_width, edgecolor=visual_map['severless_cost_128_batch_size']['edgecolor'], hatch=visual_map['severless_cost_128_batch_size']['hatch'], label='Serverless Cost (128 batch size)', alpha=visual_map['severless_cost_128_batch_size']['alpha'])
plt.bar(r4, severless_cost_256, color=visual_map['severless_cost_256_batch_size']['color'], width=bar_width, edgecolor=visual_map['severless_cost_256_batch_size']['edgecolor'], hatch=visual_map['severless_cost_256_batch_size']['hatch'], label='Serverless Cost (256 batch size)', alpha=visual_map['severless_cost_256_batch_size']['alpha'])

# Adding values above the bars
for r, cost in zip(r1, aws_redis_cost):
    plt.text(r, cost + 20, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)

for r, cost in zip(r2, severless_cost_8):
    plt.text(r, cost + 20, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)

for r, cost in zip(r3, severless_cost_128):
    plt.text(r, cost + 20, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)

for r, cost in zip(r4, severless_cost_256):
    plt.text(r, cost + 20, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)

# Adding titles and labels
plt.xlabel('% Dataset Cached', fontsize=12)
plt.ylabel('Cost ($)', fontsize=12)
plt.xticks([r + bar_width*1.5 for r in range(len(percentages))], percentages, fontsize=10)
plt.legend()
plt.title('Comparison of Costs')
plt.tight_layout()
plt.tight_layout()
plt.savefig(f'figures/cost_analysis/batch_size_image_on_cost.png', bbox_inches='tight')

plt.show()