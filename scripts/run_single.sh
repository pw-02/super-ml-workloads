#!/bin/bash
set -e

workload="imagenet_resnet50" # Define your workload
dataloder="super" # Define your dataloader
gpu_indices=(0) # Define an array of GPU indices

current_datetime=$(date +"%Y-%m-%d_%H-%M-%S") # Get the current date and time

# Define experiment ID
expid="single_job_$current_datetime"

root_log_dir="logs"
log_dir="$root_log_dir/$workload/$dataloder/$expid"

s3_bucket_for_exports="supercloudwtachexports"

echo "Cleaning up CloudWatch logs for experiment $expid"
python aws_utils/cleanup_cloudwatchlogs_for_experiment.py

# Start the resource monitor in the background
echo "Starting Resource Monitor..."
nohup python mlworkloads/resource_monitor.py start --interval 1 --flush_interval 10 --file_path "$log_dir/resource_usage_metrics.json" &
monitor_pid=$!

# Loop over each job
job_pids=()  # Array to hold the process IDs of the image classifier jobs
training_started_datetime=$(date -u +"%Y-%m-%d_%H-%M-%S") # Get the current UTC date and time
echo "Training started UTC Time: $training_started_datetime"

for gpu_index in "${gpu_indices[@]}"; do
    echo "Starting job on GPU $gpu_index with exp_id $expid"
    CUDA_VISIBLE_DEVICES="$gpu_index" python mlworkloads/run.py workload="$workload" exp_id="$expid" job_id="$gpu_index" dataloader="$dataloder" &
    job_pids+=($!)  # Save the PID of the background job
    sleep 2  # Adjust as necessary
done

# Wait for all image classifier jobs to finish
for pid in "${job_pids[@]}"; do
    wait $pid
done
training_ended_datetime=$(date -u +"%Y-%m-%d_%H-%M-%S") # Get the current UTC date and time
echo "Training ended UTC Time: $training_ended_datetime"

# Stop the resource monitor
echo "Stopping Resource Monitor..."
kill $monitor_pid

# # Download CloudWatch logs
# echo "Downloading CloudWatch logs to $log_dir"
# python aws_utils/get_cloudwatchlogs_for_experiment.py --download_dir "$log_dir" --s3_bucket_name "$s3_bucket_for_exports" --start_time "$training_started_datetime" --end_time "$training_ended_datetime"

echo "Experiment completed."
