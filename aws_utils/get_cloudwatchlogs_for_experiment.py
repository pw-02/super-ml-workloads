import boto3
import os
import time
import argparse

def export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix):
    logs_client = boto3.client('logs')

    """Exports CloudWatch logs to S3."""
    # Create a unique export task name
    task_name = f"export-{log_group_name}-{int(time.time())}"

    # Define the export task
    response = logs_client.create_export_task(
        taskName=task_name,
        logGroupName=log_group_name,
        fromTime=0,  # Export logs from the past day
        to=int(time.time() * 1000),  # Export logs up to now
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
        time.sleep(5)
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

def get_cloud_watch_logs_for_experiment(download_dir, s3_bucket_name):
    logs_client = boto3.client('logs')
    os.makedirs(download_dir, exist_ok=True)
    log_groups = logs_client.describe_log_groups(logGroupNamePrefix='/')['logGroups']
    for log_group in log_groups:
        log_group_name = log_group['logGroupName']
        # Export logs to S3
        s3_prefix = f'cloudwatchlogs/{log_group_name.replace("/", "_")}'
        export_logs_to_s3(log_group_name, s3_bucket_name, s3_prefix)

        # Download logs from S3
        download_logs_from_s3(s3_bucket_name, s3_prefix, download_dir)

        # Optionally delete logs from CloudWatch
        # delete_log_group(log_group_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CloudWatch logs to S3 and download them.")
    parser.add_argument("download_dir", help="Directory to download the logs to")
    parser.add_argument("s3_bucket_name", help="S3 bucket name for exporting logs")

    args = parser.parse_args()
    get_cloud_watch_logs_for_experiment(args.download_dir, args.s3_bucket_name)
