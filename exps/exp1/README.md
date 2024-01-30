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



## Issue Log:

1. The first time the experiment was run with super it repeatedly tried to cache batches ''8406193351831191442" and "1615931262124447054". The logs for super were filled with 'INFO: Batch'{xxxx}' is cached or in actively being cached'. The training job did not make any progress while this was happening.

   
