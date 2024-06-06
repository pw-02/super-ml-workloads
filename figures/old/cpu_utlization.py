import matplotlib.pyplot as plt
import numpy as np

# Dummy data (replace with your actual data)
time_seconds = np.linspace(0, 2500, num=300)  # Increase the number of data points for smoother curve

# CPU utilization for each data loader
cpu_utilization_pytorch = np.interp(time_seconds, [0, 1000, 2500], [10, 85, 85])  # Dummy CPU utilization values for PyTorch
cpu_utilization_shade = np.interp(time_seconds, [0, 1000, 2500], [10, 80, 80])    # Dummy CPU utilization values for Shade
cpu_utilization_super = np.interp(time_seconds, [0, 1000, 2500], [5, 20, 20])    # Dummy CPU utilization values for Super


# Add random noise to CPU utilization values
noise_level = 5  # Adjust the noise level as needed
cpu_utilization_pytorch += np.random.uniform(-noise_level, noise_level, size=len(time_seconds))
cpu_utilization_shade += np.random.uniform(-noise_level, noise_level, size=len(time_seconds))
cpu_utilization_super += np.random.uniform(-noise_level, noise_level, size=len(time_seconds))

# Define colors for each data loader
pytorch_color ='#9A9AA0'  
shade_color = '#64866A'   
super_color = '#0B3041'    

# Plotting CPU utilization for each data loader
plt.figure(figsize=(4.5, 4))
plt.plot(time_seconds, cpu_utilization_pytorch, label='PyTorch', color=pytorch_color, linestyle='-', alpha=0.7)
plt.plot(time_seconds, cpu_utilization_shade, label='Shade', color=shade_color, linestyle='-', alpha=0.7)
plt.plot(time_seconds, cpu_utilization_super, label='Super', color=super_color, linestyle='-', alpha=0.7)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('CPU Utilization (%)', fontsize=12 )


# Reduce white space between bars
plt.subplots_adjust(left=0.162, right=0.9, top=0.9, bottom=0.126)
plt.xticks(fontsize=12, weight='normal')  # Adjust the font size as needed

# Adjust legend position
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
# plt.legend()

# plt.show()


plt.savefig('figures/cpu_utilization/resnet18_cpu.png')
