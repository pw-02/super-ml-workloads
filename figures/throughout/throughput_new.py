import matplotlib.pyplot as plt

# Data
data = {
    'PyTorch': 65,
    'SHADE': 80,
    'SUPER': 185,
    # 'Oracle': 210
}


pytorch_color ='#F1F1F2'  
shade_color = '#B2C6B6'   
super_color = '#0B3041'    

pytorch_pattern ='|'  
shade_pattern = '/'   
super_pattern = '.'    # Maroon

# Extract labels and values
labels = list(data.keys())
values = list(data.values())

# Plotting the bar chart
plt.figure(figsize=(4.5, 4))
bar = plt.bar(labels,
               values, color=[pytorch_color, shade_color, super_color], 
        edgecolor='black',
        linewidth=1.5,
        # hatch=[pytorch_pattern, shade_pattern, super_pattern], 
        alpha=1, width=0.5 )

# Reduce white space between bars
plt.subplots_adjust(left=0.162, right=0.9, top=0.9, bottom=0.1)

# Adding labels and title
plt.ylabel('Mini-batches/Second', fontsize=12)

# Set font size for bar labels
plt.xticks(fontsize=12, weight='normal')  # Adjust the font size as needed

# # Adding legend
# plt.legend(bar, data.keys())

# Adding legend
# plt.legend(bar, data.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data))

# # Show plot
# plt.show()
plt.savefig('figures/throughout/resnet18_tp.png')

# Create a separate figure for the legend
fig_legend = plt.figure(figsize=(4.5, 1))
plt.figlegend(bar, data.keys(), loc='center', ncol=len(data))

# Remove axes
plt.axis('off')

# Save the legend figure
plt.savefig('figures/throughout/legend.png')


# plt.savefig('figures/throughout/resnet18_tp.png')
pass