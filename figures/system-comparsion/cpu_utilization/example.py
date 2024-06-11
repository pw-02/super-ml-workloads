import numpy as np
import matplotlib.pyplot as plt

# Number of data points
num_data_points = 1000

# Generate dummy data for CPU utilization (between 0 and 100) for three different scenarios
cpu_utilization_scenario1 = np.random.randint(0, 101, size=num_data_points)
cpu_utilization_scenario2 = np.random.randint(0, 101, size=num_data_points)
cpu_utilization_scenario3 = np.random.randint(0, 101, size=num_data_points)

# Sort the data for each scenario
sorted_cpu_utilization_scenario1 = np.sort(cpu_utilization_scenario1)
sorted_cpu_utilization_scenario2 = np.sort(cpu_utilization_scenario2)
sorted_cpu_utilization_scenario3 = np.sort(cpu_utilization_scenario3)

# Calculate the cumulative percentage for each scenario
cumulative_percentage = np.linspace(0.0, 100.0, num_data_points)

# Plot the CDF for each scenario
plt.plot(sorted_cpu_utilization_scenario1, cumulative_percentage, label='Scenario 1')
plt.plot(sorted_cpu_utilization_scenario2, cumulative_percentage, label='Scenario 2')
plt.plot(sorted_cpu_utilization_scenario3, cumulative_percentage, label='Scenario 3')

# Add labels, title, and legend
plt.xlabel('CPU Utilization')
plt.ylabel('Cumulative Percentage')
plt.title('Cumulative Distribution Function (CDF) of CPU Utilization')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
