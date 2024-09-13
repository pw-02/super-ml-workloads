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
            metrics["total_bathces"] += len(csv_data["Batch Index"])
            metrics["total_samples"] += sum(csv_data["Batch Size"])
            metrics["total_time(s)"] += sum(csv_data["Iteration Time (s)"])
            metrics["wait_on_data_time(s)"] += sum(csv_data["Wait for Data Time (s)"])
            metrics["gpu_processing_time(s)"] += sum(csv_data["GPU Processing Time (s))"])
            metrics["data_loading_time(s)"] += sum(csv_data["Data Load Time (s)"])
            metrics["transformation_time(s)"] += sum(csv_data["Transformation Time (s)"])
            metrics["cache_hits"] += sum(csv_data["Cache_Hits (Samples)"])

    #         data_times = csv_data["data_time"]
    #         transform_times = csv_data["transform_time"]

    #         for idx, value in enumerate(data_times):
    #             if value >=1:
    #                 metrics["transform_time(s)"] += transform_times[idx]
    #             pass

    # metrics["data_time(s)"] = metrics["data_time(s)"] - metrics["transform_time(s)"]


    for key in ['total_time(s)',"wait_on_data_time(s)", "gpu_processing_time(s)","data_loading_time(s)", "transformation_time(s)" ]:
        metrics[key] = metrics[key] / metrics['num_jobs']
    
    metrics["throughput(batches/s)"] = metrics["total_bathces"]/metrics["total_time(s)"]
    metrics["throughput(samples/s)"] = metrics["total_samples"]/metrics["total_time(s)"]
    metrics["cache_hit%"] = metrics["cache_hits"]/metrics["total_samples"]
    metrics["compute_time(%)"] = metrics["gpu_processing_time(s)"]/metrics["total_time(s)"]
    metrics["waiting_on_data_time(%)"] = metrics["wait_on_data_time(s)"]/metrics["total_time(s)"]
    metrics["transform_time(%)"] = metrics["transform_time(s)"]/(metrics["transformation_time(s)"]+metrics["data_loading_time(s)"])
    metrics["data_loading_time(%)"] = metrics["data_loading_time(s)"]/(metrics["transformation_time(s)"]+metrics["data_loading_time(s)"])
    return metrics

def save_dict_to_csv(data_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)


if __name__ == "__main__":
    folder_path = "C:\\Users\\pw\\Desktop\\experiment_results\\logs\\cifar10_resnet18"
    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\imagenet1k\\resnet50"

    subfolders = glob.glob(os.path.join(folder_path, '*'))
    for subfolder in subfolders:
        csv_data = convert_all_csv_to_dict(subfolder)
        folder_name = os.path.basename(os.path.normpath(subfolder))

        output_file = os.path.join(subfolder, f'{folder_name}_summary.csv')
        save_dict_to_csv(csv_data, output_file)
        print(csv_data)
