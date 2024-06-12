import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    'resnet18': {
        'Baseline': {'IO %': 95.6666846, 'Transformation %': 1.8661447000000002, 'GPU %': 2.4641960000000003},
        'Shade': {'IO %': 85.3959751, 'Transformation %': 5.1151575, 'GPU %': 9.4888674},
        'Super': {'IO %': 75.1497567, 'Transformation %': 3.4210784000000003, 'GPU %': 21.4043124}
    },
    'resnet50': {
        'Baseline': {'IO %': 63.487930999999996, 'Transformation %': 12.5521935, 'GPU %': 23.9567849},
        'Shade': {'IO %': 62.0729296, 'Transformation %': 12.5968416, 'GPU %': 25.330228700000003},
        'Super': {'IO %': 32.918037, 'Transformation %': 15.4180125, 'GPU %': 51.6578348}
    },
    'pythia-14m': {
        'Baseline': {'IO %': 23.362813, 'Transformation %': 22.9424676, 'GPU %': 53.5314123},
        'Super': {'IO %': 0.20316990000000001, 'Transformation %': 0.5530769, 'GPU %': 98.9407284}
    },
    'pythia-70m': {
        'Baseline': {'IO %': 12.4290025, 'Transformation %': 12.225082299999999, 'GPU %': 74.8647018},
        'Super': {'IO %': 0.0930038, 'Transformation %': 0.3872712, 'GPU %': 98.8831299}
    }
}

categories = list(data.keys())
subcategories = list(set([key for category in data.values() for key in category.keys()]))
metrics = ['IO %', 'Transformation %', 'GPU %']
colors = ['blue', 'orange', 'green', 'red']

fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs = axs.flatten()

for idx, category in enumerate(categories):
    ax = axs[idx]
    x = np.arange(len(metrics))
    width = 0.2  # Width of the bars
    
    for i, subcat in enumerate(subcategories):
        y = [data[category][subcat][metric] if subcat in data[category] else 0 for metric in metrics]
        ax.bar(x + i * width, y, width, label=subcat, color=colors[i % len(colors)])
    
    ax.set_title(category)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage')
    ax.set_xticks(x + width * (len(subcategories) / 2))
    ax.set_xticklabels(metrics)
    ax.legend(title='Subcategories')
    ax.grid(True)

plt.tight_layout()
plt.show()
