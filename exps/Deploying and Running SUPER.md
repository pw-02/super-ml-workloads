# Deploying and Running SUPER

There are two sperate solutions that need to be deployed in order to test SUPER

## SUPER GRPC Server

1. Ensure docker is installed and run the following to download the SUPER GRPC Server Image

   ```
   docker pull pwatters991/super-grpc:1.0
   ```

2. Next run an instance of the container, passing in the AWS CLI configurations as environment variables

```
sudo docker run -it --rm \
  -e AWS_ACCESS_KEY_ID={ENTER KEY HERE} \
  -e AWS_SECRET_ACCESS_KEY={ENTER KEY HER} \
  -e AWS_DEFAULT_REGION={ENTER KEY HER} \
  -p 50051:50051 \
  pwatters991/super-grpc:1.0
```

3. Start the server

   ```
   python server.py
   ```

## SUPER Example DL Worklaods

1. Ensure docker is installed and run the following to download the SUPER Example Workloads Image. Note its is recommended to run the workloads on an EC2 with GPU support. See other file for information on how to do this.

```
docker pull pwatters991/super-mlworloads:1.0
```

2. Next run an instance of the container, passing in the AWS CLI configurations as environment variables.

   ```
   docker run --gpus all -it --rm \
     -e AWS_ACCESS_KEY_ID={ENTER KEY HERE} \
     -e AWS_SECRET_ACCESS_KEY={ENTER KEY HERE} \
     -e AWS_DEFAULT_REGION={ENTER KEY HERE} \
     pwatters991/super-mlworloads:1.0
   ```

3. Start the training job

   ```
   python classification/launch.py
   ```