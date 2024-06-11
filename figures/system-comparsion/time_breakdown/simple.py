import matplotlib.pyplot as plt

# Data
tasks = ['I/O', 'Transform', 'GPU Compute']
times = [15, 25, 60]  # in percentage
colors = ['lightblue', 'lightgreen', 'lightcoral']

# Plot
plt.figure(figsize=(8, 6))
plt.bar(tasks, times, color=colors[0])
bottom = times
for i in range(1, len(tasks)):
    plt.bar(tasks, times, bottom=bottom, color=colors[i])
    bottom = [sum(x) for x in zip(bottom, times)]

plt.xlabel('Tasks')
plt.ylabel('Percentage of Time Spent')
plt.title('Time Breakdown in Deep Learning Training Job')
plt.legend(tasks)
plt.ylim(0, 100)  # Set y-axis limit to 0-100
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
