# #!/bin/bash

# # Start the first Python script and run it in the background


# python mlworklaods/run.py config.training.learning_rate=0.01

# # Start the second Python script and run it in the background
# # python mlworklaods/image_classification/launch.py &

# # Wait for all background processes to finish
# wait


#!/bin/bash
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")
expid="single_job_$current_datetime"

# Define an array of GPU indices
gpu_indices=(0)

# Define an array of learning rates
learning_rates=(0.001)

# Loop over each job
for i in "${!gpu_indices[@]}"; do
    echo "$expid"

    gpu_index=${gpu_indices[$i]}
    lr=${learning_rates[$i]}
    CUDA_VISIBLE_DEVICES="$gpu_index" python mlworklaods/run.py training.learning_rate="$lr" exp_id="$expid" job_id="$gpu_index" &
    sleep 1  # Wait for 1 second
done

# Wait for all background jobs to finish
wait
