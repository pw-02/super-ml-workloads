import matplotlib.pyplot as plt
import numpy as np
from storage_costs_calculations import system_comaprsion

# Assuming data is fetched using your main function
data = system_comaprsion()

visual_map = {
    'aws_redis_cost': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
    'elasticache_serverless_cost': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1},
    'severless_cost': {'color': '#4C8BB8', 'hatch': '-', 'edgecolor': 'black', 'alpha': 1},
}


# Extract data for plotting
percentages = [f"{entry['percentage']/1000:.0f}K" for entry in data]

# percentages = [entry['percentage'] for entry in data]
severless_cost = [entry['severless_cost'] for entry in data]
aws_redis_cost = [entry['aws_redis_cost'] for entry in data]
elasticache_serverless_cost = [entry['elasticache_serverless_cost'] for entry in data]

# Bar width and positions
bar_width = 0.35
r1 = np.arange(len(percentages))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plotting
plt.figure(figsize=(8.6, 3.8))


# plt.bar(r3, elasticache_serverless_cost, color='#007E7E', width=bar_width, edgecolor='black', hatch = '/', label='ElastiCache Serverless Cost',alpha=1)
plt.bar(r2, severless_cost, color='#4C8BB8', width=bar_width, edgecolor='black', hatch='-', label='Serverless Cost', alpha=0.8)
plt.bar(r1, aws_redis_cost, color='#FEA400', width=bar_width, edgecolor='black', hatch='.', label='AWS Redis Cost', alpha=0.8)

# Appending values outside the bars
for r, cost in zip(r1, aws_redis_cost):
    plt.text(r - bar_width/2, cost + 5, f'${cost:.2f}', fontsize=9, ha='center', va='bottom')

for r, cost in zip(r2, severless_cost):
    plt.text(r - bar_width/2, cost + 5, f' ${cost:.2f}', fontsize=9, ha='left', va='bottom')

# Adding trend line
plt.plot(r2, severless_cost,  linestyle='-', color='#4C8BB8', linewidth=1.3,alpha=0.8)
plt.plot(r1, aws_redis_cost,  linestyle='-', color='#FEA400', linewidth=1.3,alpha=0.8)
# plt.plot(r3, elasticache_serverless_cost,  linestyle='-', color='#007E7E', linewidth=1)

# Adding titles and labels
plt.xlabel('Number of Minibatches', fontsize=12)
plt.ylabel('Cost ($)', fontsize=12)
plt.xticks([r + bar_width for r in range(len(percentages))], percentages, fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/cost_analysis/1month_storage_cost.png', bbox_inches='tight')
plt.show()
