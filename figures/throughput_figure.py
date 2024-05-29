import matplotlib.pyplot as plt

data = {
    'Baseline': 4.13688431219181,
    'SHADE': 13.8218243663058,
    'Oracle': 68.6356083667513,
    'SUPER': 42.5324474846663,
}

pytorch_color ='#F1F1F2'  
shade_color = '#767171'
oracle_color = '#406474'   
super_color = '#B2C6B6'    

pytorch_pattern ='|'  
shade_pattern = '/'   
super_pattern = '.'    # Maroon

# Extract labels and values
labels = list(data.keys())
values = list(data.values())

# Plotting the bar chart
plt.figure(figsize=(4.5, 4))
bar = plt.bar(
    labels,
    values, color=[pytorch_color, shade_color,oracle_color, super_color], 
    edgecolor='black',
    linewidth=1.5,
    alpha=1, width=0.5 
    )

# Reduce white space between bars
plt.subplots_adjust(left=0.162, right=0.9, top=0.9, bottom=0.1)

# Adding labels and title
plt.ylabel('Mini-batches/Second', fontsize=12)

# Set font size for bar labels
plt.xticks(fontsize=12, weight='normal')  # Adjust the font size as needed

# # Adding legend
plt.legend(bar, data.keys())

# plt.show()
plt.savefig('figures/eval_resnet18_cifar10/throughput.png')
