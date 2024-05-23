import glob
import pandas as pd
import os
from collections import OrderedDict
import csv
def convert_csv_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='list')

def convert_all_csv_to_dict(folder_path):
    metrics = OrderedDict({
         "num_jobs": 0,
         "elapsed_time(s)":1000,
         "aggregated_time(s)": 0,
         "aggregated_batches": 0,
         "aggregated_compute_time(s)": 0,
         "aggregated_data_time(s)": 0,
         "aggregated_transform_time(s)": 0,
         "aggregated_cache_hits": 0
         })

    csv_data = {}
    search_pattern = os.path.join(folder_path, '**', '*.csv')
    for csv_file in glob.iglob(search_pattern, recursive=True):
        file_name = os.path.relpath(csv_file, folder_path)
        if 'metrics.csv' in file_name:
            csv_data = convert_csv_to_dict(csv_file)
            metrics["num_jobs"] +=1
            metrics["aggregated_time(s)"] += sum(csv_data["batch_time"])
            metrics["aggregated_batches"] += len(csv_data["batch_idx"])
            metrics["aggregated_compute_time(s)"] += sum(csv_data["compute_time"])
            metrics["aggregated_data_time(s)"] += sum(csv_data["data_time"])
            metrics["aggregated_transform_time(s)"] += sum(csv_data["transform_time"])
            metrics["aggregated_cache_hits"] += (sum(csv_data["cache_hits"])//csv_data["batch_size"][0])
        
    metrics["aggregated_data_time(s)"] = metrics["aggregated_data_time(s)"] -  metrics["aggregated_transform_time(s)"]
    metrics["throughput(batches_per_second)"] = metrics["aggregated_batches"]/metrics["elapsed_time(s)"]
    metrics["cache_hit%"] = metrics["aggregated_cache_hits"]/metrics["aggregated_batches"]
    metrics["compute%"] = metrics["aggregated_compute_time(s)"]/metrics["aggregated_time(s)"]
    metrics["data%"] = metrics["aggregated_data_time(s)"]/metrics["aggregated_time(s)"]
    metrics["transform%"] = metrics["aggregated_transform_time(s)"]/metrics["aggregated_time(s)"]

    return metrics

def save_dict_to_csv(data_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)


if __name__ == "__main__":
    folder_path = "C:\\Users\\pw\\Desktop\\logs\\cifar10\\resnet18"

    subfolders = glob.glob(os.path.join(folder_path, '*'))
    for subfolder in subfolders:
        csv_data = convert_all_csv_to_dict(subfolder)
        output_file = os.path.join(subfolder, "summary.csv")
        csv_data = convert_all_csv_to_dict(subfolder)
        save_dict_to_csv(csv_data, output_file)
        print(csv_data)
