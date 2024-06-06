import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
experiments = ['Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4']
io_time = [20, 25, 30.25]  # % of time spent on I/O
transformation_time = [40, 35, 30.25]  # % of time spent on data transformation
gpu_time = [40, 40, 40,50]  # % of time spent on GPU computations

# Plotting
bar_width = 0.25
index = np.arange(len(experiments))

plt.figure(figsize=(10, 6))
plt.bar(index, io_time, bar_width, label='I/O')
plt.bar(index + bar_width, transformation_time, bar_width, label='Data Transformation')
plt.bar(index + 2 * bar_width, gpu_time, bar_width, label='GPU Computations')

plt.xlabel('Experiments')
plt.ylabel('% of Time Spent')
plt.title('Distribution of Time Spent on Different Components')
plt.xticks(index + bar_width, experiments)
plt.legend()
plt.tight_layout()
plt.show()
