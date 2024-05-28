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
         "total_bathces":0,
         "total_samples":0,
         "total_time(s)": 0,
         "data_time(s)": 0,
         "compute_time(s)": 0,
         "transform_time(s)": 0,
         "cache_hits": 0,
         })

    csv_data = {}
    search_pattern = os.path.join(folder_path, '**', '*.csv')
    for csv_file in glob.iglob(search_pattern, recursive=True):
        file_name = os.path.relpath(csv_file, folder_path)
        if 'metrics.csv' in file_name:
            csv_data = convert_csv_to_dict(csv_file)
            metrics["num_jobs"] +=1
            metrics["total_bathces"] += len(csv_data["batch_idx"])
            metrics["total_samples"] += sum(csv_data["batch_size"])
            metrics["total_time(s)"] += sum(csv_data["batch_time"])
            # metrics["total_time(s)"] += csv_data["elapsed_time(s)"][len(csv_data["elapsed_time(s)"])-1]
            metrics["data_time(s)"] += sum(csv_data["data_time"])
            metrics["compute_time(s)"] += sum(csv_data["compute_time"])
            # metrics["transform_time(s)"] += sum(csv_data["transform_time"])
            metrics["cache_hits"] += sum(csv_data["cache_hits"])

            data_times = csv_data["data_time"]
            transform_times = csv_data["transform_time"]

            for idx, value in enumerate(data_times):
                if value >=1:
                    metrics["transform_time(s)"] += transform_times[idx]
                pass

    metrics["data_time(s)"] = metrics["data_time(s)"] - metrics["transform_time(s)"]


    for key in ['total_time(s)',"data_time(s)", "compute_time(s)","transform_time(s)" ]:
        metrics[key] = metrics[key] / metrics['num_jobs']
    
    metrics["throughput(batches_per_second)"] = metrics["total_bathces"]/metrics["total_time(s)"]
    metrics["cache_hit%"] = metrics["cache_hits"]/metrics["total_samples"]
    metrics["compute%"] = metrics["compute_time(s)"]/metrics["total_time(s)"]
    metrics["data%"] = metrics["data_time(s)"]/metrics["total_time(s)"]
    metrics["transform%"] = metrics["transform_time(s)"]/metrics["total_time(s)"]

    return metrics

def save_dict_to_csv(data_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)


if __name__ == "__main__":
    folder_path = "C:\\Users\\pw\\Desktop\\logs\\cifar10\\resnet18"
    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\imagenet1k\\resnet50"

    subfolders = glob.glob(os.path.join(folder_path, '*'))
    for subfolder in subfolders:
        csv_data = convert_all_csv_to_dict(subfolder)
        folder_name = os.path.basename(os.path.normpath(subfolder))

        output_file = os.path.join(subfolder, f'{folder_name}_summary.csv')
        save_dict_to_csv(csv_data, output_file)
        print(csv_data)
