# Setting up a GPU on an AWS EC2 Instance

To use a GPU on an EC2 instance that has a GPU a driver for the GPU must be installed. There are two options.

## Option 1 - Use AMI

Drivers are preinstalled in the following AMI "AWS Deep Learning Base GPU AMI (Ubuntu 20.04)". So, when deploying an EC2 instance, select this AMI and everything should work out of the box. More information here:  https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/ 

## Option 2 - Use AMI

You can install the drivers yourself on the instance using the following: 

```
#!/bin/bash

# Update package lists
sudo apt update

# Install necessary packages
sudo apt install -y build-essential gcc wget

# Download and install CUDA 12.1.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run

# Display installation logs
cat /var/log/cuda-installer.log
cat /var/log/nvidia-installer.log

# Set environment variables for CUDA 12.1
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Verify CUDA installation
nvcc --version
nvidia-smi

# Open ~/.profile in VI for editing
vi ~/.profile

# Add the following lines to ~/.profile

echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile

# Display updated ~/.profile
cat ~/.profile

# Display GPU info
nvidia-smi

```

