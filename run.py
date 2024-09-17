# #!/bin/bash

# # Define experiment configurations
# workloads=('imagenet_resnet50' 'cifar10_resnet18' 'laora_finetine_owt')
# dataloaders=('shade' 'coordl' 'litdata')
# max_parallel_jobs=4  # Adjust this value based on your system resources

# # Run experiments
# for workload in "${workloads[@]}"; do
#   for dataloader in "${dataloaders[@]}"; do
#     echo "Running experiment: $workload, $dataloader"
#     ./run_experiment.sh "$workload" "$dataloader" &
    
#     # Limit parallel jobs
#     while [ $(jobs -r | wc -l) -ge $max_parallel_jobs ]; do
#       wait -n
#     done
#   done
# done


# ./run_experiment.sh imagenet_resnet50 shade