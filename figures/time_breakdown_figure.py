import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# # Data resnet 18
# data = {
#     'Basline': {'IO %': 96.5325224804555, 'Transformation %': 1.0398589536327, 'GPU %': 2.42467468291663},
#     'SHADE': {'IO %': 87.1375296699801, 'Transformation %': 7.22306871099934, 'GPU %': 5.63940161902051},
#     'Oracle': {'IO %': 24, 'Transformation %': 3, 'GPU %': 74},
#     'SUPER': {'IO %': 38, 'Transformation %': 14, 'GPU %': 48},
# }


# Data resnet 50
data = {
    'Basline': {'IO %': 79.1254446154995, 'Transformation %': 7.1445961114448, 'GPU %': 13.728382949341},
    'SHADE': {'IO %': 60.5862122174053, 'Transformation %': 12.7536208474465, 'GPU %': 26.6601669351481},
    'Oracle': {'IO %': 12.9087571589205, 'Transformation %': 13.5274180063709, 'GPU %': 73.5560405113968},
    'SUPER': {'IO %': 36.759908419515, 'Transformation %': 14.9225592790088, 'GPU %': 48.3118074995643},
}


# # Data resnet18
# data = {
#     'Basline': {'IO %': 96, 'Transformation %': 1, 'GPU %': 2},
#     'SHADE': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
#     'Oracle': {'IO %': 66, 'Transformation %': 1, 'GPU %': 33},
#     'SUPER': {'IO %': 77, 'Transformation %': 1, 'GPU %': 22},
# }

# # Data pythia-14m
# data = {
#     'Basline': {'IO %': 33.9185439956913, 'Transformation %': 31.9787106787411, 'GPU %': 34.1027453258919},
# #     'SHADE': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
#     'Oracle': {'IO %': 0.485495741989118, 'Transformation %': 0.229738774945173, 'GPU %': 99.2847654817468},
#     'SUPER': {'IO %': 0.545340337925344, 'Transformation %': 1.81757997119794, 'GPU %': 97.6370796923387},
# }

# # Data pythia-70m
# data = {
#     'Basline': {'IO %': 21.738547697084, 'Transformation %': 21.3326087949571, 'GPU %': 56.9288435074687},
# #    'SHADE': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
#     'Oracle': {'IO %': 0.183879063850989, 'Transformation %': 0.53540833086722, 'GPU %': 98.9196007792605},
#     'SUPER': {'IO %': 0.31788307853497, 'Transformation %':0.89652015639459, 'GPU %': 99.5411007793089},
# }

io_color ='#F1F1F2'  
# shade_color = '#767171'
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

plt.savefig('figures\eval_resnet50_cifar10\percentage_breakdown.png')
