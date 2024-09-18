import boto3
import os
import time
import argparse
from datetime import datetime
from datetime import datetime, timezone
import concurrent.futures

def export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix, start_time_str, end_time = None):

    start_time_ms = convert_to_milliseconds(start_time_str)
    # end_time_ms = int(time.time() * 1000)  # Current time in milliseconds
    if end_time:
        end_time_ms = convert_to_milliseconds(end_time)
    else:
        end_time_ms = int(time.time() * 1000)
    logs_client = boto3.client('logs')

    """Exports CloudWatch logs to S3."""
    # Create a unique export task name
    task_name = f"export-{log_group_name}-{int(time.time())}"
    
    # Define the export task
    response = logs_client.create_export_task(
        taskName=task_name,
        logGroupName=log_group_name,
        fromTime=start_time_ms,  # Export logs from the past day
        to=end_time_ms,  # Export logs up to now
        destination=s3_bucket_name,
        destinationPrefix=s3_prefix
    )

    task_id = response['taskId']
    # Monitor the status of the export task
    while True:
        request = logs_client.describe_export_tasks(taskId=task_id)
        status = request['exportTasks'][0]['status']['code']
        print(f'Task ID {task_id} status: {status}')

        if status in ['COMPLETED', 'FAILED']:
            break  
        # Wait for a while before checking the status again
        time.sleep(10)
    print(f"Export task created: {response['taskId']} for log group {log_group_name}")

def download_logs_from_s3(s3_bucket_name, s3_prefix, download_path):
    s3_client = boto3.client('s3')

    """Downloads exported logs from S3."""
    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
    if 'Contents' not in response:
        print(f"No logs found in S3 bucket {s3_bucket_name} with prefix {s3_prefix}.")
        return
    
    # Iterate over each file in the S3 prefix
    for obj in response['Contents']:
        # Get the S3 object key
        s3_key = obj['Key']
        
        # Create corresponding local path
        local_path = os.path.join(download_path, *s3_key.split('/'))
        local_path = os.path.normpath(local_path)  # Normalize path for Windows

        # Ensure the local directory exists
        local_dirname = os.path.dirname(local_path)
        os.makedirs(local_dirname, exist_ok=True)
        
        # Download the S3 object to the local path
        s3_client.download_file(s3_bucket_name, s3_key, local_path)

def get_cloud_watch_logs_for_experiment(download_dir, s3_bucket_name, start_time_str, end_time_str=None):
    logs_client = boto3.client('logs')
    os.makedirs(download_dir, exist_ok=True)
    log_groups = logs_client.describe_log_groups(logGroupNamePrefix='/')['logGroups']
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Export logs in parallel
        futures = []
        # for log_group in log_groups:
        #     log_group_name = log_group['logGroupName']
        #     s3_prefix = f'cloudwatchlogs/{log_group_name.replace("/", "_")}'
        #     futures.append(executor.submit(export_logs_to_s3, log_group_name, s3_bucket_name, s3_prefix, start_time_str, end_time_str))
        
        # # Wait for all export tasks to complete
        # for future in concurrent.futures.as_completed(futures):
        #     try:
        #         future.result()  # re-raise any exceptions that occurred during export
        #     except Exception as e:
        #         print(f'Exception during log export: {e}')

        # Download logs from S3 in parallel
        for log_group in log_groups:
            log_group_name = log_group['logGroupName']
            s3_prefix = f'cloudwatchlogs/{log_group_name.replace("/", "_")}'
            executor.submit(download_logs_from_s3, s3_bucket_name, s3_prefix, download_dir)
            
        # Wait for all download tasks to complete
        # (You may need to use additional synchronization here depending on your needs)



# def get_cloud_watch_logs_for_experiment(download_dir, s3_bucket_name, start_time_str, end_time_str=None):
#     logs_client = boto3.client('logs')
#     os.makedirs(download_dir, exist_ok=True)
#     log_groups = logs_client.describe_log_groups(logGroupNamePrefix='/')['logGroups']
#     for log_group in log_groups:
#         log_group_name = log_group['logGroupName']
#         # Export logs to S3
#         s3_prefix = f'cloudwatchlogs/{log_group_name.replace("/", "_")}'
#         export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix, start_time_str, end_time_str)

#         # Download logs from S3
    
#     download_logs_from_s3(s3_bucket_name, s3_prefix, download_dir)



def convert_to_milliseconds(date_str):
    # Parse the input string as a naive datetime object
    dt = datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')
    
    # Make the datetime object timezone-aware (UTC)
    dt_utc = dt.replace(tzinfo=timezone.utc)
    
    # Convert to milliseconds since the Unix epoch
    return int(dt_utc.timestamp() * 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CloudWatch logs to S3 and download them.")
    parser.add_argument("--download_dir", help="Directory to download the logs to", default="logs")
    parser.add_argument("--s3_bucket_name", help="S3 bucket name for exporting logs", default="supercloudwtachexports")
    parser.add_argument("--start_time", help="", default='2024-09-16_20-03-03')
    parser.add_argument("--end_time", help="",  default='2024-09-17_20-03-03')

    args = parser.parse_args()
    get_cloud_watch_logs_for_experiment(args.download_dir, args.s3_bucket_name, args.start_time, args.end_time)

