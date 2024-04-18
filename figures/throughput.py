import matplotlib.pyplot as plt
import numpy as np

# Data for PyTorch
pytorch_batches = [100, 95, 80, 75, 68, 33, 55 ]  # Replace these values with your actual data

# Data for Shade
shade_batches = [90, 85, 70, 65, 23, 87, 81]  # Replace these values with your actual data

# Data for Super
super_batches = [110, 105, 90, 85, 99, 60, 75]  # Replace these values with your actual data

models = ['ShuffleNet', 'AlexNet', 'ResNet18', 'SqueezeNet', 'ResNet50', 'Pythia70m', 'Pythia160m']

# Define bar width and positions
bar_width = 0.25
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Custom color choices
pytorch_color ='#C8620E'  
shade_color = '#005A9B'   
super_color = '#b30000'    # Maroon

# Plotting the bar graph with customized colors and patterns
plt.bar(r1, pytorch_batches, color=pytorch_color, width=bar_width, edgecolor='black', label='PyTorch', hatch='/', alpha=0.7)
plt.bar(r2, shade_batches, color=shade_color, width=bar_width, edgecolor='black', label='SHADE', hatch='.', alpha=0.7)
plt.bar(r3, super_batches, color=super_color, width=bar_width, edgecolor='black', label='SUPER', hatch='-', alpha=0.7)

# Adding labels and titles
# plt.xlabel('Models')
plt.ylabel('Batches/Second', fontsize=12)
# plt.title('Comparison of Data Loaders')
plt.xticks([r + bar_width for r in range(len(models))], models, fontsize=12)
plt.legend()
# Increase font size of y-axis values
plt.tick_params(axis='x', labelsize=12)
# Increase font size of y-axis values
plt.tick_params(axis='y', labelsize=12)
# Make "Super" label bold in the legend
leg = plt.legend()
leg.get_texts()[2].set_fontweight('bold')

# Show plot
# plt.tight_layout()
plt.show()






print()