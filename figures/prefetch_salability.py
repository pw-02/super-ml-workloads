import matplotlib.pyplot as plt
import pandas as pd

# Dummy data
data = {
    'file_size': ['2MB']*5 + ['50MB']*5 + ['500MB']*5 + ['1GB']*5 + ['5GB']*5,
    'concurrency': [4, 8, 16, 32, 48] * 5,
    'throughput': [120, 230, 340, 430, 500, 90, 170, 240, 300, 350, 40, 75, 100, 120, 135, 30, 55, 75, 90, 100, 5, 9, 12, 15, 18],
    # 'avg_latency': [0.1, 0.12, 0.14, 0.15, 0.17, 0.2, 0.25, 0.28, 0.32, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 2.5, 3.0, 3.5, 4.0, 4.5]
}

colors = [
    '#767171', '#767171', '#406474', '#B2C6B6', '#B2C6B6', '#B2C6B6'
]


# Convert to DataFrame
df = pd.DataFrame(data)

# Plot throughput for different file sizes
# plt.figure(figsize=(14, 7))
plt.figure(figsize=(4.5, 4))

for i, file_size in enumerate(df['file_size'].unique()):
    subset = df[df['file_size'] == file_size]
    plt.plot(subset['concurrency'], subset['throughput'], marker='o', label=f'{file_size}', color=colors[i])

# plt.title('Throughput vs Num Workers for Different File Sizes')
plt.xlabel('Number Workers')
plt.ylabel('Throughput (requests per second)')
plt.legend()
plt.grid(True)
plt.show()

# # Plot latency for different file sizes
# plt.figure(figsize=(14, 7))
# for file_size in df['file_size'].unique():
#     subset = df[df['file_size'] == file_size]
#     plt.plot(subset['concurrency'], subset['avg_latency'], marker='o', label=f'{file_size}')

# plt.title('Latency vs Concurrency for Different File Sizes')
# plt.xlabel('Concurrency Level')
# plt.ylabel('Average Latency (seconds)')
# plt.legend()
# plt.grid(True)
# plt.show()

pass
print()