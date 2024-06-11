import glob
import pandas as pd
import os
from collections import OrderedDict
import csv
def convert_csv_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='list')

def process_reprot(folder_path, is_language, folder_name):
    metrics = OrderedDict({
        "datalaoder": '',
         "num_jobs": 0,
        #  "elapsed_time(s)": 0,
         "total_bathces":0,
         "total_samples":0,
         "total_time(s)": 0,
         "compute_time(s)": 0,
         "dataloading_time(s)": 0,
         "fetch_time(s)": 0,
         "transform_time(s)": 0,
         "dataloading_delay_time(s)": 0,
         "io_delay(s)": 0,
         "transform_delay(s)": 0,
         "cache_hits": 0,
         "throughput(batches_per_second)": 0,
         "compute%": 0,
         "io%": 0,
         "transform%": 0,
         "cache_hit%": 0,
         })

    csv_data = {}
    search_pattern = os.path.join(folder_path, '**', '*.csv')
    for csv_file in glob.iglob(search_pattern, recursive=True):
        file_name = os.path.relpath(csv_file, folder_path)

        if 'metrics.csv' in file_name:
            metrics["datalaoder"] = folder_name.split('_multi_job')[0]
            csv_data = convert_csv_to_dict(csv_file)
            metrics["num_jobs"] +=1
            # metrics["elapsed_time(s)"] += csv_data["elapsed_time"][-1]
            if 'batch_idx' in csv_data:
                metrics["total_bathces"] += len(csv_data["batch_idx"])
            else:
                metrics["total_bathces"] += len(csv_data["iter"])
            
            if 'batch_size' in csv_data:
                metrics["total_samples"] += sum(csv_data["batch_size"])
            else:
                metrics["total_samples"] += sum(csv_data["samples"])
            metrics["total_time(s)"] += sum(csv_data["batch_time"])
            metrics["compute_time(s)"] += sum(csv_data["compute_time"])
            if 'shade'in folder_path:
                metrics["dataloading_delay_time(s)"] = metrics["total_time(s)"]-metrics["compute_time(s)"]
            else:
                metrics["dataloading_delay_time(s)"] += sum(csv_data["data_time"])
            metrics["fetch_time(s)"] += sum(csv_data["fetch_time"])
            metrics["transform_time(s)"] += sum(csv_data["transform_time"])
            metrics["cache_hits"] += sum(csv_data["cache_hits"])
    
    metrics["dataloading_time(s)"] = metrics["fetch_time(s)"] + metrics["transform_time(s)"]
    
    for key in ['total_time(s)',"dataloading_delay_time(s)", "fetch_time(s)","dataloading_time(s)","compute_time(s)","transform_time(s)" ]:
        metrics[key] = metrics[key] / metrics['num_jobs']
    # for key in ['total_time(s)',"data_time(s)", "compute_time(s)","transform_time(s)" ]:
    #     metrics[key] = metrics[key] / metrics['num_jobs']
    
    metrics["throughput(batches_per_second)"] = metrics["total_bathces"]/metrics["total_time(s)"]
    if is_language:
        metrics["cache_hit%"] = metrics["cache_hits"]/metrics["total_bathces"]
    else:
        metrics["cache_hit%"] = metrics["cache_hits"]/metrics["total_samples"]

    percntage_of_data_loading_time_spent_on_io = metrics["fetch_time(s)"]/metrics["dataloading_time(s)"]
    percntage_of_data_loading_time_spent_on_trans = metrics["transform_time(s)"]/metrics["dataloading_time(s)"]

    metrics["io_delay(s)"] = metrics["dataloading_delay_time(s)"]*percntage_of_data_loading_time_spent_on_io
    metrics["transform_delay(s)"] = metrics["dataloading_delay_time(s)"]*percntage_of_data_loading_time_spent_on_trans
    metrics["compute%"] = metrics["compute_time(s)"]/metrics["total_time(s)"]
    metrics["io%"] =  metrics["io_delay(s)"]/metrics["total_time(s)"]
    metrics["transform%"] = metrics["transform_delay(s)"]/metrics["total_time(s)"]

    return metrics


def save_dict_to_csv(data_dict, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
        writer.writeheader()
        writer.writerow(data_dict)

if __name__ == "__main__":

    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\cifar10\\resnet18"
    # folder_path = "C:\\Users\\pw\\Desktop\\logs\\imagenet1k\\resnet50"
    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\openwebtext\\pythia-14m"
    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\openwebtext\\pythia-70m"
    #folder_path = "C:\\Users\\pw\\Desktop\\logs\\openwebtext\\pythia-160m"
    #folder_path = "C:\\Users\\pw\\Desktop\\full_hits\\cifar10\\resnet18"

    folder_paths = [
         "C:\\Users\\pw\\Desktop\\reports\\cifar10\\resnet18",
         "C:\\Users\\pw\\Desktop\\reports\\imagenet1k\\resnet50",
        "C:\\Users\\pw\\Desktop\\reports\\openwebtext\\pythia-14m",
        "C:\\Users\\pw\\Desktop\\reports\\openwebtext\\pythia-70m",
        "C:\\Users\\pw\\Desktop\\reports\\openwebtext\\pythia-160m",

    ]
    combined_files_paths = []

    for folder_path  in folder_paths:
        combined_data = []
        all_items = glob.glob(os.path.join(folder_path, '*'))
        subfolders = [item for item in all_items if os.path.isdir(item)]

        for subfolder in subfolders:
            folder_name = os.path.basename(os.path.normpath(subfolder))
            if 'openwebtext' in folder_path:
                csv_data = process_reprot(subfolder,is_language=True, folder_name=folder_name)
            else:
                csv_data = process_reprot(subfolder,is_language=False, folder_name=folder_name)

            combined_data.append(csv_data)

            output_file = os.path.join(subfolder, f'{folder_name}_summary.csv')
            save_dict_to_csv(csv_data, output_file)
            print(csv_data)
        
        base_name = os.path.basename(os.path.normpath(folder_path))
        with open(os.path.join(folder_path, f'{base_name}_dataloader_comparsion.csv'), 'w', newline='') as csvfile:
            combined_files_paths.append(os.path.join(folder_path, f'{base_name}_dataloader_comparsion.csv'))
            writer = csv.DictWriter(csvfile, fieldnames=combined_data[0].keys())
            writer.writeheader()
            for i in combined_data:
                writer.writerow(i)


dataframes = []
# Read each CSV file and append the DataFrame to the list
for file in combined_files_paths:

    # Get the parent folder name
    parent_folder_name = os.path.basename(os.path.dirname(file))
    df = pd.read_csv(file)
    df.insert(0, 'workload', parent_folder_name)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('C:\\Users\\pw\\Desktop\\reports\\summary.csv', index=False)

print('All files have been merged into merged_output.csv')
