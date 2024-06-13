import matplotlib.pyplot as plt
from storage_costs_calculations import impact_of_differnt_batch_sizes
import numpy as np

data = impact_of_differnt_batch_sizes()

# Extracting batch_sizes and costs
batch_sizes = [entry['batch_size'] for entry in data]
costs = [entry['cost'] for entry in data]

# Bar width and positions
bar_width = 0.6
r = np.arange(len(batch_sizes))

# Plotting
plt.figure(figsize=(8.2, 3.8))
bars = plt.bar(r, costs, color='#4C8BB8', hatch='-', width=bar_width, edgecolor='black', alpha=0.8,  label='Serverless')

# Adding a straight red line
plt.axhline(y=1651.27, color='red', linestyle='--', linewidth=2, label='AWS Redis')

# Adding labels and title
plt.xlabel('Batch Size', fontsize=10)
plt.ylabel('Cost', fontsize=10)
plt.xticks(r, batch_sizes, fontsize=10)
plt.yticks(fontsize=10)

# Adding values on top of the bars
for bar, cost in zip(bars, costs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, f'${cost:.2f}', 
             ha='center', va='bottom', fontsize=9)
plt.ylim([0, 2000])
plt.legend()
# Show pl
plt.tight_layout()
plt.savefig(f'figures/cost_analysis/batch_size_image_on_cost.png', bbox_inches='tight')

plt.show()