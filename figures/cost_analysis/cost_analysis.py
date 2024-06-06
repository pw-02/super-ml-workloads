import matplotlib.pyplot as plt
import numpy as np

# Dummy data for the graph
num_minibatches = np.arange(1, 101)

# baseline_color ='#007E7E'  
disk = '#FEA400'
redis_color = '#4C8BB8'   
super_color = '#000000' 

workload = 'resnet18'

# Generate incremental costs for three different scenarios
incremental_costs_scenario_1 = np.random.uniform(1, 3, num_minibatches.size)
incremental_costs_scenario_2 = np.random.uniform(2, 4, num_minibatches.size)
incremental_costs_scenario_3 = np.random.uniform(1.5, 3.5, num_minibatches.size)

# Calculate cumulative costs for each scenario
costs_scenario_1 = np.cumsum(incremental_costs_scenario_1)
costs_scenario_2 = np.cumsum(incremental_costs_scenario_2)
costs_scenario_3 = np.cumsum(incremental_costs_scenario_3)

plt.figure(figsize=(4, 3))

plt.plot(num_minibatches, costs_scenario_1,  linewidth=2.5, label='SUPER', color=disk)  # 'k-' for black solid line
plt.plot(num_minibatches, costs_scenario_2,  linewidth=2.5, label='Redis', color=redis_color)  # 'k-' for black solid line
plt.plot(num_minibatches, costs_scenario_3,  linewidth=2.5, label='Disk', color=super_color)  # 'k-' for black solid line

# Label the axes
plt.xlabel('# Minibatches Processed', fontsize=10)
plt.ylabel('Cost($)')

plt.subplots_adjust(left=0.16, right=0.955, top=0.9, bottom=0.167)

plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add a legend
plt.legend()

# # Show the plot
# plt.show()

plt.savefig(f'figures/cost_analysis/{workload}_cost.png')

