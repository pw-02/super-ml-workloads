# Exp 1 - Training a single classification model on ResNet18

## Description and Expected Outcomes:

The goal for this experiment is to test how my solution performs when training a single model using a single GPU. The outcome of this experiment, should highlight any issues with my solution that need to be resolved as well as being able to answer the following questions:

- Did the training fail or stop before completion, if so, what was the cause of this?
- What is the accuracy of the trained model?
- What was the cost of training?
- What was the training throughput?
- What was the data loading time?
  -  What was the data fetch time?
  -  What was the batch decoding time?
- How much time was spent processing on GPU?
- What was the GPU utilization?

## What needs to be done before I can run this experiment?

- ~~Update the reporting of the training jobs to capture the required information.~~
- ~~Create a configuration file for the experiment.~~
- Document the environment setup so that I replicate needed if I need to
- Develop a script that will track the GPU utilization.
- Confirm a cost model for calculating the cost
  - Cost of SION Proxy and GRPC Service
  - Cost of Lambda
  - Cos of training node

## Experiment Description

- Dataset: tiny-imagenet200
- Model: resnet18
- Data-loader backend: SUPER
- Epochs: 3
- GPU: 1 x NVIDIA T4 Tensor Core GPUs
- EC2 Instance: G4dn
- Pytorch Workers = 4
- Shuffle dataset = True
- Batch size: 256

# Issue Log:

## Repletely Displaying  'Batch'{xxxx}' is cached or in actively being cached'

#### Description:

The first time the experiment was run with super it repeatedly tried to cache batches ''8406193351831191442" and "1615931262124447054". The logs for super were filled with 'INFO: Batch'{xxxx}' is cached or in actively being cached'. The training job did not make any progress while this was happening.

#### Cause:

The training job encountered a cache miss, so it contacted super to get a 'status' for the batch. The 'status' was that the batch was in progress, which is true, but it was taking a while because of the cold start penalty. The training job repeatedly called the function, it was only supposed to call it a max times of 10, but there was a error in the loop which caused it run until the data was returned with a cache hit.

#### Solution:

The Super Dataset code has been fixed. Specifically, the 'max_attempts' rule in the 'fetch_from_cache' function, which wasn't doing anything previously. 

#### Resolved? 

Yes

------

## The order of batches being cached is different than order that batches are being accessed

#### Description:

All of the jobs are experiencing a cache misses, even though the SUPER data loader says that the data has been added to the cache. 

#### Cause:

The 'bucket_name' is hard coded in SUPER, and is used to generate the 'batch_id'. In this experiment the S3 bucket used is different to the one that is hard coded within SUPER.

#### Solution:

Modify SUPER so that it can handle batches from any S3 bucket and therefore there is no need to hardcode a bucket name.

#### Resolved? 

Yes

------

## Training Not Working with Super when Pytorch Workers > 4

#### Description:

When the number of pytorch workers is > 0, the training loop just stalls and doesn't process any batches. It does not crash, and I can see information being sent to the SUPER service from the sampler. 

#### Cause:

When number workers > 0 multi-processing is used. This means each dataset object will require its own SUPER client and cache, meaning I cannot use the same client for all datasets. 

#### Solution:

Change code so that every super dataset object creates its own instance of a super client and a cache_client. 

#### Resolved? 

Yes - also added some functionality to the training script that confirms access to SUPER and the Cache before starting training.

------

## SUPER incorrectly assigning batches as 'cached'

#### Description:

During the processing of batches, SUPER is calling lambda to preload the batch into the cache, however the lambda always returns 'batch cached' even when the caching has failed for some reason. This has a knock on effect as SUPER assumes the batch has been cached.

#### Cause:

Lambda function always returns 'batch cached', even when the caching of the batch failed. The batch in question here failed because of the folloiwng 

"Error: stack expects each tensor to be equal size, but got [3, 64, 64] at entry 0 and [1, 64, 64] at entry 63'"

#### Solution:

There is some 'dirty' images in the resnet data. Converting them to RNG during the processing of the images resolves the problem.

#### Resolved? 

Yes

------

## Unable to communicate with SUPER from within custom datasets when number workers > 0

When the number of pytorch workers is >0, the entire training process stalls for infinity when a call is made to super from within the custom dataset. 

#### Cause:

When number pytorch workers > 0, multiprocessing is enabled and each worker gets its own instance of the dataset class. python GRPC does not support multiprocessing, whenever a channel has already be created in the parent process (which is the case in my solution as I am also commuicating with super from within the sampler which is in the main (parent) process of the datasets (children). The issue is explained here: and example code to reproduce the issue is below:

```
import multiprocessing
import time
def worker(process_id, server_address):
    # Each worker creates its own instance of CacheCoordinatorClient
    client = SuperClient(server_address)
    while True:     
        # Example: Call GetBatchStatus RPC
        response = client.get_batch_status(process_id, 'example_dataset')
        print(f"Process {process_id}: GetBatchStatus Response - {response}")
        time.sleep(2)

    # Close the gRPC channel when the worker is done
    #del client



def run():
     # Example of using the CacheCoordinatorClient with multiple processes
    server_address = 'localhost:50051'
    num_processes = 4

    # Create a list to hold the process objects
    processes = []

    # Create a super client for the main process
    main_client  = SuperClient(server_address)
    # Example: Call additional RPC with the super client
    
    response = main_client.get_batch_status(123, 'example_dataset')
    print(f"Process Main: GetBatchStatus Response - {response}")
    
    # Fork child processes
    for i in range(0, num_processes):
        process = multiprocessing.Process(target=worker, args=(i, server_address))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
   
```



Note: that issue only occurs when the channel is created in the main, before being created in the child. If we moved the 'main_client' in the example above, to after the initialization of the child workers the example would work. However, this simple workaround doesn't translate over to well PyTorch (at least in my tests), and the issue stills occurs regardless. The other caveat in creating a client per process is that number of clients the sever can support is limited, so we want to be as efficient as possible. For instances, if we had 4 workers + 1 main thread, and 10 training jobs, that is a total of 50 connection stubs. If we only use the connection in the main process this will reduce to 10. Another option that would work is to enforce the number of pytorch workers to be 0, but this may break existing scripts and cause performance issues elsewhere as training and preloading are being performed in the once process. 

#### Solution:

In the end, the solution I went for it to avoid creating super_clients in the custom datasets. Instead create a single client in the main, and using it in the main for all the communication that I need. The benefits of this approach are that is it is cleaner, easier to manage,and reduces the number of connections to the server allowing it to support more training jobs concurrently. The one downside is that i cannot commmunciate to the server from within the datasets anymore, so any information that i want to pass between them must be done outside. This is slightly annoying, because I only want to communciate with Super in the case of cache miss during the data fetch, but I don't know if I will experience a cache miss or not beforehand. Therefore I have to account for all eventualities, and check the cache status of each batch prior to feeding them to the dataset, as part of the sampler. This is a cost of an extra message to the grpc server for every batch being processed. The overhead of doing this needs measured. The logic may change later to reduce the number of requests somehow.

#### Resolved? 

Yes 
