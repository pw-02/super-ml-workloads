import matplotlib.pyplot as plt
import numpy as np

# Generate increasing dummy data for demonstration
num_epochs = 50
base_accuracy_your_loader = 0.5
base_accuracy_pytorch_loader = 0.6

# Increase the accuracy over time
accuracy_your_loader = np.linspace(base_accuracy_your_loader, 0.9, num_epochs)
accuracy_pytorch_loader = np.linspace(base_accuracy_pytorch_loader, 0.92, num_epochs)

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), accuracy_your_loader, label='Your Data Loader')
plt.plot(range(num_epochs), accuracy_pytorch_loader, label='PyTorch Data Loader')
plt.xlabel('Epochs')
plt.ylabel('Top-1 Accuracy')
plt.title('Comparison of Top-1 Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.show()
