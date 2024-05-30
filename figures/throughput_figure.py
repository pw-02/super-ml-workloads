import matplotlib.pyplot as plt
# # resnet18
# data = {
#     'Baseline': 4.13688431219181,
#     'SHADE': 13.8218243663058,
#     'Oracle': 83.6356083667513,
#     'SUPER': 42.5324474846663,
# }

#resnet50
data = {
    'Baseline': 1.61999563618888,
    'SHADE': 3.13318390260257,
    'Oracle': 8.66404492084878,
    'SUPER': 5.65832064020547,
}
# #pythia 14m
# data = {
#     'Baseline': 5.19032196005167,
#     'Oracle': 15.0729524232023,
#     'SUPER': 14.6204069615893,
# }


# #pythia 70m
# data = {
#     'Baseline': 6.032028159589,
#     'Oracle': 9.87844349124468,
#     'SUPER': 10.5352221118991,
# }


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
    values, color=[pytorch_color,shade_color,oracle_color, super_color], 
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
plt.savefig('figures/eval_resnet50_cifar10/throughput.png')
