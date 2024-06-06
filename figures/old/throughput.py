import matplotlib.pyplot as plt
import numpy as np

# Data for PyTorch
shade_batches = [13.8218243663058, 3.13318390260257, 5.19032196005167,  6.032028159589]  # Replace these values with your actual data
# Data for Shade
pytorch_batches = [4.13688431219181, 1.61999563618888, 5.19032196005167, 6.032028159589]  # Replace these values with your actual data
# Data for Super
oracle_batches = [83.6356083667513,  8.66404492084878, 15.0729524232023, 10.5352221118991]  # Replace these values with your actual data

super_batches = [42.5324474846663, 5.65832064020547, 14.6204069615893, 9.87844349124468]  # Replace these values with your actual data

models = ['ResNet-18/Cifar10', 'ResNet-50/ImageNet', 'Pythia-14m/OpenWebText', 'Pythia-70m/OpenWebText']

# Define bar width and positions
bar_width = 0.2
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# # Custom color choices
baseline_color ='#007E7E'  
sota_color = '#FEA400'
oracle_color = '#4C8BB8'   
super_color = '#000000'    
plt.figure(figsize=(12, 4))
# Plotting the bar graph with customized colors and patterns
plt.bar(r1, pytorch_batches, color=baseline_color, width=bar_width, edgecolor='black', label='PyTorch', hatch='/')
plt.bar(r2, shade_batches, color=sota_color, width=bar_width, edgecolor='black', label='SHADE', hatch='.')
plt.bar(r3, oracle_batches, color=oracle_color, width=bar_width, edgecolor='black', label='SUPER', hatch='-')
plt.bar(r4, super_batches, color=super_color, width=bar_width, edgecolor='black', label='SUPER', hatch='-')

# Adding labels and titles
# plt.xlabel('Models')
plt.ylabel('Aggregated minibatches/second', fontsize=12)
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

plt.savefig('test.png')



