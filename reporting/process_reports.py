import glob
import pandas as pd
import os
from collections import OrderedDict
import csv

def convert_csv_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='list')


# def cacluate_redis_serverless_cost(duartion, cache_size_gb, request_rate, size_per_request):
#     # Duration is in seconds
#     # Memory size is in GB
#     # Cost is in USD
#     hours_in_a_month = 730
#     data_storage_cost_monthly = cache_size_gb * hours_in_a_month * 0.125

#     data_transfer_cost_monthly = 0
#     cache_size_kb = cache_size_gb * 1024 * 1024
    




def convert_all_csv_to_dict_text(folder_path):
    folder_name = os.path.basename(os.path.normpath(subfolder))
    metrics = OrderedDict({
         "name": folder_name,
         "num_jobs": 0,
         "total_batches": 0,
         "total_samples": 0,
         "total_time(s)": 0,
         "wait_on_data_time(s)": 0,
         "gpu_processing_time(s)": 0,
         "data_loading_time(s)": 0,
         "transformation_time(s)": 0,
         "cache_hits": 0,
    })

    search_pattern = os.path.join(folder_path, '**', '*.csv')
    for csv_file in glob.iglob(search_pattern, recursive=True):
        file_name = os.path.relpath(csv_file, folder_path)
        if 'metrics.csv' in file_name:
            csv_data = convert_csv_to_dict(csv_file)
            metrics["num_jobs"] += 1
            metrics["total_batches"] += len(csv_data["Batch Index"])
            metrics["total_tokens"] += sum(csv_data["Batch Size (Tokens)"])
            metrics["total_time(s)"] += sum(csv_data["Iteration Time (s)"])
            metrics["wait_on_data_time(s)"] += sum(csv_data["Iteration Time (s)"]) - sum(csv_data["GPU Processing Time (s)"])
            metrics["gpu_processing_time(s)"] += sum(csv_data["GPU Processing Time (s)"])
            metrics["data_loading_time(s)"] += sum(csv_data["Data Load Time (s)"])
            metrics["transformation_time(s)"] += sum(csv_data["Transformation Time (s)"])
            metrics["cache_hits"] += sum(csv_data["Cache_Hit (Batch)"])

    if metrics['num_jobs'] > 0:
        for key in ['total_time(s)', "wait_on_data_time(s)", "gpu_processing_time(s)", "data_loading_time(s)", "transformation_time(s)"]:
            metrics[key] = metrics[key] / metrics['num_jobs']
        
        metrics["throughput(batches/s)"] = metrics["total_batches"] / metrics["total_time(s)"]
        metrics["throughput(tokens/s)"] = metrics["total_samples"] / metrics["total_time(s)"]
        metrics["cache_hit(%)"] = metrics["cache_hits"] / metrics["total_batches"]
        metrics["compute_time(%)"] = metrics["gpu_processing_time(s)"] / metrics["total_time(s)"]
        metrics["waiting_on_data_time(%)"] = metrics["wait_on_data_time(s)"] / metrics["total_time(s)"]
        metrics["transform_time(%)"] = metrics["transformation_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_loading_time(s)"])
        metrics["data_loading_time(%)"] = metrics["data_loading_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_loading_time(s)"])
    
    return metrics



def convert_all_csv_to_dict_vision(folder_path):
    folder_name = os.path.basename(os.path.normpath(subfolder))
    metrics = OrderedDict({
        "name": folder_name,
         "num_jobs": 0,
         "total_batches": 0,
         "total_samples": 0,
         "total_time(s)": 0,
         "wait_on_data_time(s)": 0,
         "gpu_processing_time(s)": 0,
         "data_loading_time(s)": 0,
         "transformation_time(s)": 0,
         "cache_hits": 0,
    })

    search_pattern = os.path.join(folder_path, '**', '*.csv')
    for csv_file in glob.iglob(search_pattern, recursive=True):
        file_name = os.path.relpath(csv_file, folder_path)
        if 'metrics.csv' in file_name:
            csv_data = convert_csv_to_dict(csv_file)
            metrics["num_jobs"] += 1
            metrics["total_batches"] += len(csv_data["Batch Index"])
            metrics["total_samples"] += sum(csv_data["Batch Size"])
            metrics["total_time(s)"] += sum(csv_data["Iteration Time (s)"])
            metrics["wait_on_data_time(s)"] += sum(csv_data["Iteration Time (s)"]) - sum(csv_data["GPU Processing Time (s)"])
            metrics["gpu_processing_time(s)"] += sum(csv_data["GPU Processing Time (s)"])
            metrics["data_loading_time(s)"] += sum(csv_data["Data Load Time (s)"])
            metrics["transformation_time(s)"] += sum(csv_data["Transformation Time (s)"])
            metrics["cache_hits"] += sum(csv_data["Cache_Hits (Samples)"])

    if metrics['num_jobs'] > 0:
        for key in ['total_time(s)', "wait_on_data_time(s)", "gpu_processing_time(s)", "data_loading_time(s)", "transformation_time(s)"]:
            metrics[key] = metrics[key] / metrics['num_jobs']
        
        metrics["throughput(batches/s)"] = metrics["total_batches"] / metrics["total_time(s)"]
        metrics["throughput(samples/s)"] = metrics["total_samples"] / metrics["total_time(s)"]
        metrics["cache_hit(%)"] = metrics["cache_hits"] / metrics["total_batches"]
        metrics["compute_time(%)"] = metrics["gpu_processing_time(s)"] / metrics["total_time(s)"]
        metrics["waiting_on_data_time(%)"] = metrics["wait_on_data_time(s)"] / metrics["total_time(s)"]
        metrics["transform_time(%)"] = metrics["transformation_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_loading_time(s)"])
        metrics["data_loading_time(%)"] = metrics["data_loading_time(s)"] / (metrics["transformation_time(s)"] + metrics["data_loading_time(s)"])
    
    return metrics

def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    
    # Get all unique fieldnames from all dictionaries
    fieldnames = set().union(*(d.keys() for d in dict_list))

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in dict_list:
            writer.writerow(data)

if __name__ == "__main__":
    folder_path = "C:\\Users\\pw\\Desktop\\dataloading_gpu_unlimited-cache_results\\imagenet_resnet50\coordl"
    subfolders = glob.glob(os.path.join(folder_path, '*'))
    
    overall_summary = []

    for subfolder in subfolders:
        csv_data = convert_all_csv_to_dict_vision(subfolder)
        #csv_data = convert_all_csv_to_dict_vision(subfolder)
        folder_name = os.path.basename(os.path.normpath(subfolder))
        overall_summary.append(csv_data)  # Collect the folder summary
        print(csv_data)

    # Save all folder summaries to a single CSV file
    save_dict_list_to_csv(overall_summary, os.path.join(folder_path, 'overall_summary.csv'))
