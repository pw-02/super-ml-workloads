import boto3

def delete_all_cloudwatch_log_groups():
    logs_client = boto3.client('logs')
    log_groups = logs_client.describe_log_groups(logGroupNamePrefix='/')['logGroups']
    for log_group in log_groups:
        log_group_name = log_group['logGroupName']
        logs_client.delete_log_group(logGroupName=log_group_name)


def empty_s3_bucket(bucket_name):
    s3_client = boto3.client('s3')
    
    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    
    if 'Contents' not in response:
        print(f"No objects found in bucket '{bucket_name}'.")
        return
    
    # Delete all objects
    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
    
    # Delete the objects
    if objects_to_delete:
        print(f"Deleting {len(objects_to_delete)} objects from bucket '{bucket_name}'.")
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={
                'Objects': objects_to_delete,
                'Quiet': True
            }
        )
    
    # Check if there are more objects to delete (pagination)
    while response.get('IsTruncated'):  # Continue if there are more pages
        continuation_token = response.get('NextContinuationToken')
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            ContinuationToken=continuation_token
        )
        
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        if objects_to_delete:
            print(f"Deleting {len(objects_to_delete)} more objects from bucket '{bucket_name}'.")
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={
                    'Objects': objects_to_delete,
                    'Quiet': True
                }
            )
    
def cleanup_cloudwtachlogs_for_experiment():
    delete_all_cloudwatch_log_groups()
    empty_s3_bucket("supercloudwtachexports")


if __name__ == "__main__":
    cleanup_cloudwtachlogs_for_experiment()
