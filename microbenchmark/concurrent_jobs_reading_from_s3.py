import boto3
import time
import concurrent.futures
import random
import matplotlib.pyplot as plt
import numpy as np
from urllib.parse import urlparse

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()

# Initialize S3 client
s3_client = boto3.client('s3')
s3_path = 's3://imagenet1k-sdl/train/'
bucket_name = S3Url(s3_path).bucket
prefix = S3Url(s3_path).key
num_files = 1000  # Number of files to read
  # replace with your S3 bucket name
import json

def _get_sample_list_from_s3(s3_prefix, s3_bucket, use_index_file=True, images_only=True, max_dataset_size = None):
        
        keys = []

        if max_dataset_size:
            index_file_key = f"{s3_prefix}_paired_index_{max_dataset_size}GB.json"
        else:
            index_file_key = f"{s3_prefix}_paired_index.json"

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)

                for class_name, samples in paired_samples.items():
                    if len(keys) >= num_files:
                        break
                    for sample in samples:
                        keys.append(sample)
                        if len(keys) >= num_files:
                            break
                return keys
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")

        paginator = s3_client.get_paginator('list_objects_v2')
        total_size_gb = 0

        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
            if max_dataset_size and total_size_gb >= max_dataset_size:
                    break
            for blob in page.get('Contents', []):
                if max_dataset_size and total_size_gb >= max_dataset_size:
                    break
                blob_path = blob.get('Key')
                
                if blob_path.endswith("/"):
                    continue  # Skip folders
                
                stripped_path = blob_path[len(s3_prefix):].lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip

                if images_only and not blob_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue  # Skip index file

                keys.append(blob_path)
                total_size_gb += blob['Size'] / 1024 / 1024 / 1024

        # if use_index_file and paired_samples:
        #     s3_client.put_object(
        #         Bucket=self.s3_bucket,
        #         Key=index_file_key,
        #         Body=json.dumps(paired_samples, indent=4).encode('utf-8')
        #     )

        return paired_samples
    





def read_from_s3(file_key):
    """reading a file from S3 and measures the time taken."""
    start_time = time.time()
    s3file = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    size_mb = s3file['ContentLength'] / 1024 / 1024
    end_time = time.time()
    
    latency = end_time - start_time
    return latency, size_mb

def run_concurrent_reads(num_jobs, file_keys):
    """Runs multiple concurrent reads and returns throughput and latency."""
    latencies = []
    total_file_sizes = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
        # Each worker processes all keys to simulate intensive access
        futures = [executor.submit(read_from_s3, file_key) for _ in range(num_jobs) for file_key in file_keys]

        # Wait for all jobs to complete and record latencies
        for future in concurrent.futures.as_completed(futures):
            latency, file_size = future.result()
            latencies.append(latency)
            total_file_sizes.append(file_size)

    # Calculate throughput (MB/s) and average latency
    total_data_read = sum(total_file_sizes)
    throughput = total_data_read / sum(latencies)  # MB per second
    avg_latency = np.mean(latencies)  # Average latency per read

    return throughput, avg_latency, total_data_read

# def run_concurrent_reads(num_jobs, file_keys):
#     """Runs multiple concurrent reads and returns throughput and latency."""
#     random.shuffle(file_keys)  # Random selection of files to simulate random access
#     latencies = []
#     total_file_sizes = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
#         # Submit read jobs to the executor
#         futures = [executor.submit(read_from_s3, file_key) for file_key in file_keys]
        
#         # Wait for all jobs to complete and record latencies
#         for future in concurrent.futures.as_completed(futures):
#             latency, file_size = future.result()
#             latencies.append(latency)
#             total_file_sizes.append(file_size)
    
#     # Calculate throughput (MB/s) and average latency
#     total_data_read = sum(total_file_sizes)  # Assuming average file size is 100MB (replace with actual sizes)
#     throughput = total_data_read / (sum(latencies))  # MB per second
#     avg_latency = np.mean(latencies)  # Average latency per read

#     return throughput, avg_latency, total_data_read


# def load_file_keys(bucket, prefix): 
#     """Loads the keys of all files in an S3 bucket with a given prefix."""
#     keys = []
#     paginator = s3.get_paginator('list_objects_v2')
#     for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
#         if 'Contents' in result:
#             for key in result['Contents']:
#                 keys.append(key['Key'])
#     return keys

# Prepare file to log results
output_file = "s3_performance_results.txt"

with open(output_file, "w") as file:
    file.write("S3 Performance Results\n")
    file.write("Concurrency Level\tThroughput (MB/s)\tAvg Latency (s)\Toal Size (Mb)\n")


def main():
    # Experiment to test different levels of concurrency
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    throughputs = []
    latencies = []
    total_data_read = []

    file_keys =  _get_sample_list_from_s3(prefix, bucket_name)

    for num_jobs in concurrency_levels:
        throughput, avg_latency,total_data = run_concurrent_reads(num_jobs, file_keys)
        throughputs.append(throughput)
        latencies.append(avg_latency)
        total_data_read.append(total_data)
        print(f"Concurrency {num_jobs}: Throughput = {throughput:.2f} MB/s, Avg Latency = {avg_latency:.4f} s, Total Data Read = {total_data:.4f} MB")
         # Log to file
        with open(output_file, "a") as file:
            file.write(f"{num_jobs}\t\t\t{throughput:.2f}\t\t\t{avg_latency:.4f}\t\t\t{total_data:.4f}\n")

if __name__ == "__main__":
    main()




# # Plot the results
# fig, ax1 = plt.subplots()

# # Plot throughput (MB/s) on the left y-axis
# ax1.set_xlabel('Number of Concurrent Jobs')
# ax1.set_ylabel('Throughput (MB/s)', color='tab:blue')
# ax1.plot(concurrency_levels, throughputs, color='tab:blue', label='Throughput')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # Create a second y-axis to plot latency (s)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Average Latency (s)', color='tab:red')
# ax2.plot(concurrency_levels, latencies, color='tab:red', label='Latency', linestyle='--')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# fig.tight_layout()
# plt.title('S3 Performance under Varying Concurrency')
# plt.show()
