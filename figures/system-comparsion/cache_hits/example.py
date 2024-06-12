import matplotlib.pyplot as plt
import random

total_samples = 256000 // 128

# Generate cache hits data for three scenarios
scenarios = [
    {'cache_hits': 100000 // 128},
    {'cache_hits': 150000 // 128},
    {'cache_hits': 200000 // 128}
]

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
for scenario in scenarios:
    cache_hits = scenario['cache_hits']
    cache_data = [0] * total_samples
    cache_indices = random.sample(range(total_samples), cache_hits)
    for idx in cache_indices:
        cache_data[idx] = 1
    cumulative_cache_hits = [0] * total_samples
    for i in range(1, total_samples):
        cumulative_cache_hits[i] = cumulative_cache_hits[i - 1] + cache_data[i]
    plt.plot(cumulative_cache_hits, label=f'{cache_hits} Cache Hits')
plt.xlabel('Sample Index')
plt.ylabel('Cumulative Cache Hits')
plt.title('Cumulative Cache Hits vs Sample Index')
plt.xlim(0, total_samples)
plt.ylim(0, total_samples)  # Adjust ylim if needed
plt.legend()
plt.grid(True)
plt.show()
