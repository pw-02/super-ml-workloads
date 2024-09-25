import boto3

# Initialize the S3 client
s3 = boto3.client('s3')

# Function to list all objects in the S3 bucket
def list_files_in_s3(bucket_name, prefix=''):
    files = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    while response.get('Contents'):
        for obj in response['Contents']:
            files.append(obj['Key'])
        if response['IsTruncated']:  # If there are more files to retrieve
            continuation_token = response['NextContinuationToken']
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            break
    return files

# Read the contents of each file
def read_s3_file(bucket_name, key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return obj['Body'].read().decode('utf-8')  # Assuming the content is UTF-8 encoded

# Example usage
bucket_name = 'coco-dataset'
prefix = 'train2014/' 
file_keys = list_files_in_s3(bucket_name, prefix)
import json
import os
samples = json.loads(open('data\coco_train.json', 'r').read())

paired_samples = {}
for idx, sample in enumerate(samples):
     image_path = sample['image']
     image_id = sample["image_id"]
     image_path =  'train2014/' + os.path.basename(image_path)
     if image_path not in file_keys:
         continue
     caption = sample['caption']
     if image_id not in paired_samples:
         paired_samples[image_id] = [(image_path, caption)]
                 
for file_key in file_keys:
    print(f"Reading file: {file_key}")
    file_contents = read_s3_file(bucket_name, file_key)
    print(file_contents)  # Do something with the file contents
