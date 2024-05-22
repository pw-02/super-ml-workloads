#!/bin/bash

# Define an array of GPU indices
gpu_indices=(0 1 2 3)

# Define an array of learning rates
learning_rates=(0.001 0.01 0.1 1.0)
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
expid="single_job_$current_datetime"

# Loop over each job
for i in "${!gpu_indices[@]}"; do
    gpu_index=${gpu_indices[$i]}
    lr=${learning_rates[$i]}
    CUDA_VISIBLE_DEVICES="$gpu_index" python ../mlworklaods/run.py training.learning_rate="$lr" exp_id="$expid" job_id="$gpu_index" &z
    sleep 1  # Wait for 1 second
done

# Wait for all background jobs to finish
wait
