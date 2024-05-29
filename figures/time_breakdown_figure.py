import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# # Data resnet 50
# data = {
#     'Basline': {'IO %': 72, 'Transformation %': 14, 'GPU %': 14},
#     'SHADE': {'IO %': 46, 'Transformation %': 23, 'GPU %': 27},
#     'Oracle': {'IO %': 24, 'Transformation %': 3, 'GPU %': 74},
#     'SUPER': {'IO %': 38, 'Transformation %': 14, 'GPU %': 48},
# }


# Data resnet18
data = {
    'Basline': {'IO %': 96, 'Transformation %': 1, 'GPU %': 2},
    'SHADE': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
    'Oracle': {'IO %': 66, 'Transformation %': 1, 'GPU %': 33},
    'SUPER': {'IO %': 77, 'Transformation %': 1, 'GPU %': 22},
}


io_color ='#F1F1F2'  
shade_color = '#767171'
transformation_color = '#406474'   
gpu_color = '#B2C6B6'    

# Extract labels and values
labels = list(data.keys())
io_values = [data[label]['IO %'] for label in labels]
transformation_values = [data[label]['Transformation %'] for label in labels]
gpu_values = [data[label]['GPU %'] for label in labels]

# Plotting the bar chart
plt.figure(figsize=(4.5, 4))

# Plotting bars for IO %
plt.bar(labels, io_values, color=io_color, label='IO %', edgecolor='black',
        linewidth=1.5, width=0.5 )
# Plotting bars for Transformation %
plt.bar(labels, transformation_values, bottom=io_values, color=transformation_color, label='Transformation %',edgecolor='black',
        linewidth=1.5, width=0.5 )
# Plotting bars for GPU %
plt.bar(labels, gpu_values, bottom=[sum(x) for x in zip(io_values, transformation_values)], color=gpu_color, label='GPU %',edgecolor='black',
        linewidth=1.5, width=0.5 )

plt.subplots_adjust(left=0.162, right=0.9, top=0.9, bottom=0.1)

# Adding labels and title
plt.ylabel('Percentage of Time (%)', fontsize=12)
plt.xticks(fontsize=12, weight='normal')  # Adjust the font size as needed
plt.legend(['I/O%', 'Transform%', 'GPU%'], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
# Adjust layout to avoid overlap
# plt.tight_layout()
# Show plot
# plt.show()
# pass
# Formatter function to add percentage sign
def percent_formatter(x, pos):
    return f'{int(x)}%'

# Apply the formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))

plt.savefig('figures\eval_resnet18_cifar10\percentage_breakdown.png')
