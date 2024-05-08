import matplotlib.pyplot as plt

# Sample data
# Replace these with your actual data
epochs = list(range(1, 31))  # Increased to 30 epochs

# Example top 1 accuracy data for each sampling method
# Replace these with your actual top 1 accuracy data
super_sampling = [70, 73, 76, 78, 80, 82, 84, 85, 86, 87, 88, 89, 89, 90, 90, 91, 92, 92, 93, 93, 94, 94, 94, 95, 95, 95, 95, 96, 96, 96]
baseline_sampling = [68, 72, 75, 78, 80, 81, 83, 84, 85, 86, 87, 88, 88, 89, 89, 90, 91, 91, 92, 92, 93, 93, 93, 94, 94, 94, 94, 95, 95, 95]
shade_sampling = [69, 71, 74, 77, 79, 81, 82, 84, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94]

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot data for each sampling method
plt.plot(epochs, baseline_sampling, label='Baseline Sampling', color='black', marker='o')
plt.plot(epochs, shade_sampling, label='SHADE Sampling', color='black', marker='s')
plt.plot(epochs, super_sampling, label='SUPER Sampling', color='black', marker='^')

# Add labels and title
plt.xlabel('Number of Epochs')
plt.ylabel('Top 1 Accuracy (%)')
# plt.title('Changes in Top 1 Accuracy over Epochs')
plt.legend()

# Display the plot
plt.show()


print()