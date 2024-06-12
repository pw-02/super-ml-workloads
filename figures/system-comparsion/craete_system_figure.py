import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from matplotlib.patches import Patch

def percent_formatter(x, pos):
    return f'{int(x)}%'

def plot_throughput(ax, labels, values, visual_map, order, bar_width=0.5, include_y_axis_label=True, fontsize =12):
    labels, values = zip(*sorted(zip(labels, values), key=lambda x: order.get(x[0].lower(), float('inf'))))
    for j, label in enumerate(labels):
        visual_attr = visual_map.get(label, {})
        ax.bar(label, values[j], width=bar_width, color=visual_attr.get('color', '#4C8BB8'), 
               edgecolor=visual_attr.get('edgecolor', 'black'), 
               hatch=visual_attr.get('hatch', None), 
               alpha=visual_attr.get('alpha', 1))
    if include_y_axis_label: 
        ax.set_ylabel('Aggregated minibatches/sec', fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

def plot_time_breakdown(ax, labels, data, visual_map, bar_width=0.25, include_y_axis_label=True, fontsize =12):
    metrics = list(data[labels[0]].keys())
    x = np.arange(len(metrics))
    for i, label in enumerate(labels):
        y = [data[label][metric] for metric in metrics]
        visual_attr = visual_map.get(label, {})
        ax.bar(x + i * bar_width, y, bar_width, label=label,
               color=visual_attr.get('color', '#4C8BB8'), 
               edgecolor=visual_attr.get('edgecolor', 'black'), 
               hatch=visual_attr.get('hatch', None), 
               alpha=visual_attr.get('alpha', 1.0))
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(metrics)
    if include_y_axis_label: 
        ax.set_ylabel('Percentage of Time (%)', fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim([0, 100])
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

def plot_utilization(ax, labels, data, visual_map, bar_width=0.25, include_y_axis_label=True, fontsize =12):
    metrics = list(data[labels[0]].keys())
    x = np.arange(len(metrics))
    for i, label in enumerate(labels):
        y = [data[label][metric] for metric in metrics]
        visual_attr = visual_map.get(label, {})
        ax.bar(x + i * bar_width, y, bar_width, label=label,
               color=visual_attr.get('color', '#4C8BB8'), 
               edgecolor=visual_attr.get('edgecolor', 'black'), 
               hatch=visual_attr.get('hatch', None), 
               alpha=visual_attr.get('alpha', 1.0))
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(metrics)
    if include_y_axis_label:
        ax.set_ylabel('Resource Utilization (%)', fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim([0, 100])
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

def main():
    # Define colors, hatches, edgecolors, and alphas for specific subcategories
    visual_map = {
        'Baseline': {'color': '#007E7E', 'hatch': '/', 'edgecolor': 'black', 'alpha': 1},
        'Shade': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
        'Litdata': {'color': '#FEA400', 'hatch': '.', 'edgecolor': 'black', 'alpha': 1},
         r'$\bf{SUPER}$': {'color': '#000000', 'hatch': '-', 'edgecolor': 'black', 'alpha': 1},
    }

    csv_file_path = 'C:\\Users\\pw\\Desktop\\reports\\summary.csv' 
    df = pd.read_csv(csv_file_path)
    workloads = df['workload'].unique()
    order = {'baseline': 1, 'shade': 2, 'litdata': 2, 'super': 3}
    fontsize = 10
    for i, workload in enumerate(workloads):
        if i == 0:
            include_y_axis_label = True
        else:
            include_y_axis_label = False
            
        # fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        fig, axs = plt.subplots(3, 1, figsize=(3.5, 7.5))

        workload_df = df[(df['workload'] == workload) & (df['datalaoder'].str.lower() != 'oracle')]
        labels = [label.capitalize() for label in workload_df['datalaoder'].tolist()]

        for i, label in enumerate(labels):
            if label.lower() == 'super':
                labels[i] = r'$\bf{SUPER}$' 
        # Throughput
        throughput_values = workload_df['throughput(batches_per_second)'].tolist()
        plot_throughput(axs[0], labels, throughput_values, visual_map, order, 0.5, include_y_axis_label,fontsize)

        # Time breakdown
        io_values = [val * 100 for val in workload_df['io%'].tolist()]
        transform_values = [val * 100 for val in workload_df['transform%'].tolist()]
        compute_values = [val * 100 for val in workload_df['compute%'].tolist()]
        data = {label: {'IO': io_values[i], 'Transform': transform_values[i], 'Compute': compute_values[i]} for i, label in enumerate(labels)}
        plot_time_breakdown(axs[1], labels, data, visual_map, 0.25, include_y_axis_label,fontsize)

        # Utilization
        cpu_values = [json.loads(val)['mean'] for val in workload_df['cpu_usge'].tolist()]
        gpu_values = [json.loads(val)['mean'] for val in workload_df['gpu_usge'].tolist()]
        data = {label: {'CPU': cpu_values[i], 'GPU': gpu_values[i]} for i, label in enumerate(labels)}
        plot_utilization(axs[2], labels, data, visual_map, 0.25, include_y_axis_label, fontsize)

        plt.tight_layout()
        plt.savefig(f'figures/system-comparsion/{workload}.png', bbox_inches='tight')
        # plt.show()
        
     # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(4.48, 0.25))
    legend_labels = list(visual_map.keys())

    handles = [Patch(color=visual_map[label]['color'], label=label,hatch=visual_map[label]['hatch'], edgecolor=visual_map[label]['edgecolor'], alpha=visual_map[label]['alpha']) for label in legend_labels]
    fig_legend.legend(handles=handles,
                 labels=legend_labels,
                 loc='center',
                 ncol=len(legend_labels),
                 bbox_to_anchor=(0.5, 0.5),
                 frameon=True)  # Remove legend frame
    plt.axis('off')
    # plt.tight_layout(pad=0.01)
    plt.savefig('figures/system-comparsion/legend.png', bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    main()
