#!/bin/bash

# Define a timeout duration
TIMEOUT_DURATION=10100

# Function to run the main part of the script
run_script() {
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
        CUDA_VISIBLE_DEVICES="$gpu_index" python mlworklaods/run.py training.learning_rate="$lr" exp_id="$expid" job_id="$gpu_index" &
        sleep 2  # Wait for 2 seconds
    done

    # Wait for all background jobs to finish
    wait
}

# Run the script with a timeout
timeout $TIMEOUT_DURATION bash -c 'run_script'