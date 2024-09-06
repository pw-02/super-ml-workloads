#!/bin/bash

# Define an array of GPU indices
gpu_indices=(0 1 2 3)

# Define an array of learning rates
learning_rates=(0.001 0.01 0.1 1.0)

# Get the current date and time
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")

# Define experiment ID
expid="multi_job_$current_datetime"

# Print experiment ID to verify
echo "Experiment ID: $expid"

# Loop over each job
for i in "${!gpu_indices[@]}"; do
    gpu_index=${gpu_indices[$i]}
    lr=${learning_rates[$i]}
    echo "Starting job on GPU $gpu_index with learning rate $lr and exp_id $expid"
    CUDA_VISIBLE_DEVICES="$gpu_index" python mlworkloads/image_classifer.py workload.learning_rate="$lr" exp_id="$expid" job_id="$gpu_index" &
    sleep 2  # Wait for 1 second
done

# Wait for all background jobs to finish
wait
