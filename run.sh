# #!/bin/bash

# # Start the first Python script and run it in the background


# python mlworklaods/run.py config.training.learning_rate=0.01

# # Start the second Python script and run it in the background
# # python mlworklaods/image_classification/launch.py &

# # Wait for all background processes to finish
# wait


#!/bin/bash

# Define an array of GPU indices
gpu_indices=(0 1 2 3)

# Define an array of learning rates
learning_rates=(0.001 0.01 0.1 1.0)

# Loop over each job
for i in "${!gpu_indices[@]}"; do
    gpu_index=${gpu_indices[$i]}
    lr=${learning_rates[$i]}
    CUDA_VISIBLE_DEVICES="$gpu_index" python mlworklaods/run.py config.training.learning_rate="$lr" &
    sleep 1  # Wait for 1 second
done

# Wait for all background jobs to finish
wait
