import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

def percent_formatter(x, pos):
    return f'{int(x)}%'

io_color ='#007E7E'  
transformation_color = '#000000' #4C8BB8
gpu_color = '#FEA400'



result_data = {
    'eval_resnet18_cifar10': {
        'Baseline': {'IO %': 96.5325224804555, 'Transformation %': 1.0398589536327, 'GPU %': 2.42467468291663},
        'SHADE': {'IO %': 87.1375296699801, 'Transformation %': 7.22306871099934, 'GPU %': 5.63940161902051},
        'Oracle': {'IO %': 24, 'Transformation %': 3, 'GPU %': 74},
        r'$\bf{SUPER}$': {'IO %': 38, 'Transformation %': 14, 'GPU %': 48},
    },
    'eval_resnet50_imagenet':  {
        'Baseline': {'IO %': 79.1254446154995, 'Transformation %': 7.1445961114448, 'GPU %': 13.728382949341},
        'SHADE': {'IO %': 60.5862122174053, 'Transformation %': 12.7536208474465, 'GPU %': 26.6601669351481},
        'Oracle': {'IO %': 12.9087571589205, 'Transformation %': 13.5274180063709, 'GPU %': 73.5560405113968},
        r'$\bf{SUPER}$': {'IO %': 36.759908419515, 'Transformation %': 14.9225592790088, 'GPU %': 48.3118074995643},
    },
    'eval_owt_pythia14m': {
        'Baseline': {'IO %': 96, 'Transformation %': 1, 'GPU %': 2},
        'LitData': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
        'Oracle': {'IO %': 66, 'Transformation %': 1, 'GPU %': 33},
        r'$\bf{SUPER}$': {'IO %': 77, 'Transformation %': 1, 'GPU %': 22},
    },
    'eval_owt_pythia70m': {
        'Baseline': {'IO %': 21.738547697084, 'Transformation %': 21.3326087949571, 'GPU %': 56.9288435074687},
        'LitData': {'IO %': 93, 'Transformation %': 1, 'GPU %': 6},
        'Oracle': {'IO %': 0.183879063850989, 'Transformation %': 0.53540833086722, 'GPU %': 98.9196007792605},
        r'$\bf{SUPER}$': {'IO %': 0.31788307853497, 'Transformation %':0.89652015639459, 'GPU %': 99.5411007793089},
    }
}


for workload, data in result_data.items():
    labels = list(data.keys())
    io_values = [data[label]['IO %'] for label in labels]
    transformation_values = [data[label]['Transformation %'] for label in labels]
    gpu_values = [data[label]['GPU %'] for label in labels]

    # Plotting the bar chart
    # plt.figure(figsize=(4.5, 4))
    plt.figure(figsize=(4.5, 3))
   
    # Plotting bars for IO %
    bars_io = plt.barh(labels, io_values, color=io_color, label='IO %', edgecolor='black', alpha=1, hatch='/',linewidth=1.2, height=0.5)
    
    # Plotting bars for Transformation %
    bars_transformation = plt.barh(labels, transformation_values, left=io_values, color=transformation_color,hatch='-', label='Transformation %', edgecolor='black', alpha=1, linewidth=1.2, height=0.5)
    
    # Plotting bars for GPU %
    left_values = [sum(x) for x in zip(io_values, transformation_values)]
    bars_gpu = plt.barh(labels, gpu_values, left=left_values, color=gpu_color, label='GPU %',  hatch='.', edgecolor='black', alpha=1, linewidth=1.2, height=0.5)
    
    # plt.subplots_adjust(left=0.19, right=0.9, top=0.9, bottom=0.145)
    plt.subplots_adjust(left=0.186, right=0.983, top=0.88, bottom=0.17)

    # Add gridlines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adding labels and title
    plt.xlabel('Percentage of Time (%)', fontsize=12)
    plt.yticks(fontsize=12)  # Adjust the font size as needed
    
    
    # Apply the formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    
#     plt.legend(['I/O%', 'Transform%', 'GPU%'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    plt.legend(['I/O%', 'Transform%', 'GPU%'])

    plt.savefig(f'figures/system-comparsion/{workload}/percentage_breakdown.png')
    plt.close()