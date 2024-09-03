#!/bin/bash
set -e

workload="cifar10_resnet18" # Define your workload
gpu_indices=(0) # Define an array of GPU indices
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S") # Get the current date and time

# Define experiment ID
expid="single_job_$current_datetime"

root_log_dir="logs"
log_dir="$root_log_dir/$workload/$expid"

s3_bucket_for_exports="supercloudwtachexports"

echo "Cleaning up CloudWatch logs for experiment $expid"
python aws_utils/cleanup_cloudwatchlogs_for_experiment.py

# Start resource monitor in the background
nohup python resource_monitor.py start --interval 1 --flush_interval 10 --file_path "$log_dir/resource_usage_metrics.json" > resource_monitor.log 2>&1 &
monitor_pid=$!

# Loop over each job
for gpu_index in "${gpu_indices[@]}"; do
    echo "Starting job on GPU $gpu_index with exp_id $expid"
    CUDA_VISIBLE_DEVICES="$gpu_index" python mlworkloads/image_classifier.py workload="$workload" exp_id="$expid" job_id="$gpu_index" &
    sleep 2  # Adjust as necessary
done

# Wait for all background jobs to finish
wait

# Stop the resource monitor
echo "Stopping Resource Monitor..."
kill $monitor_pid

# Download CloudWatch logs
echo "Downloading CloudWatch logs to $log_dir"
python aws_utils/get_cloudwatchlogs_for_experiment.py "$log_dir" "$s3_bucket_for_exports"

echo "Experiment completed."
