import matplotlib.pyplot as plt

baseline_color ='#007E7E'  
sota_color = '#FEA400'
oracle_color = '#4C8BB8'   
super_color = '#000000'    


# Sample data
# Replace these with your actual data
epochs = list(range(1, 31))  # Increased to 30 epochs

# Example top 1 accuracy data for each sampling method
# Replace these with your actual top 1 accuracy data
super_sampling = [70, 73, 76, 78, 80, 82, 84, 85, 86, 87, 88, 89, 89, 90, 90, 91, 92, 92, 93, 93, 94, 94, 94, 95, 95, 95, 95, 96, 96, 96]
baseline_sampling = [68, 72, 75, 78, 80, 81, 83, 84, 85, 86, 87, 88, 88, 89, 89, 90, 91, 91, 92, 92, 93, 93, 93, 94, 94, 94, 94, 95, 95, 95]
# shade_sampling = [69, 71, 74, 77, 79, 81, 82, 84, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94]

# Plotting the data
plt.figure(figsize=(4, 3))

# Plot data for each sampling method
plt.plot(epochs, baseline_sampling, label='Baseline Sampling', color=sota_color, marker='x')
# plt.plot(epochs, shade_sampling, label='SHADE Sampling', color='black', marker='s')
plt.plot(epochs, super_sampling, label='SUPER Sampling', color=baseline_color, marker='.')

plt.subplots_adjust(left=0.13, right=0.967, top=0.979, bottom=0.16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.xlabel('Epoch #', fontsize=10)
plt.ylabel('Top 1 Accuracy', fontsize=10)
plt.legend(fontsize=8)

# Display the plot
plt.show()
plt.savefig(f'figures/convergenge.png')
pass