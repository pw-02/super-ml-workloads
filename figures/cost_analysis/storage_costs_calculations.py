from constants import *
import json

def calculate_s3_read_request_cost(num_requests):

    return s3_get_request_cost * num_requests
    


def calculate_lambda_cost(invocation_count, execution_duration_ms, memory_allocation_mb):
    # Constants for Lambda pricing (as of 2024-06-12)
    per_invocation_cost = 0.20 / 1e6  # $0.20 per million requests
    per_duration_cost = 0.00001667  # $0.00001667 per GB-second (1 second for 1 GB)

    # Calculate total duration in GB-seconds
    total_duration_gb_seconds = (execution_duration_ms / 1000) * (memory_allocation_mb / 1024)

    # Calculate total cost
    invocation_cost = invocation_count * per_invocation_cost
    execution_cost = total_duration_gb_seconds * per_duration_cost

    total_cost = invocation_cost + execution_cost

    return total_cost

def cacluate_serverless_cache_costs(num_files, batch_size, duration_hours, keep_alive_interval_mins = 15, ec2_instance_type ='c5n.2xlarge'):
   
    num_batches = num_files // batch_size

    #cacluate initial cost of populating the cache. Invoke fucntion that loads a batch worth of data from S3 and adds it to the cache
    create_batches_lmabda_cost = calculate_lambda_cost(invocation_count=num_batches, 
                                                       execution_duration_ms=3500,
                                                       memory_allocation_mb=2048)
    s3_read_requests_costs = calculate_s3_read_request_cost(num_files)
    create_batches_lmabda_cost = create_batches_lmabda_cost + s3_read_requests_costs

    num_keep_alive_requests = (duration_hours *  60)/keep_alive_interval_mins * num_batches
    keep_alive_cost = calculate_lambda_cost(invocation_count=num_keep_alive_requests, 
                                                       execution_duration_ms=40,
                                                       memory_allocation_mb=1048)
    #cost of running ec2 instnace 
    proxy_cost = ec2_on_demand_cost_per_hour[ec2_instance_type] * duration_hours
    total_cost = create_batches_lmabda_cost + keep_alive_cost + proxy_cost
    return total_cost

def caclaute_aws_redis_cache_cost(num_files, duration_hours, data_size_gb):

    best_redis_instance = find_redis_node(data_size_gb)

    cache_cost = best_redis_instance['cost_per_hour_usd'] * duration_hours
    s3_read_requests_costs = calculate_s3_read_request_cost(num_files)
    return cache_cost  + s3_read_requests_costs

def caclaute_aws_elasticache_serverless_cache_cost(data_size_gb, duration_hours):
    gb_hours = data_size_gb * duration_hours
    return gb_hours * elasticache_serverless_prices_per_gb_hour


def calculate_ebs_cost(volume_size_gb, provisioned_iops, provisioned_throughput_mbps, snapshot_size_gb, snapshot_count, instance_hours=730, storage_cost_per_gb=0.08, iops_cost_per_iops=0.005, throughput_cost_per_mbps=0.08, snapshot_cost_per_gb=0.05):
    # Constants
    baseline_iops = 3000
    baseline_throughput_mbps = 125
    hours_per_month = 730

    # Calculate instance months
    instance_months = instance_hours / hours_per_month

    # Storage cost
    storage_cost = volume_size_gb * instance_months * storage_cost_per_gb

    # IOPS cost
    billable_iops = max(0, provisioned_iops - baseline_iops)
    iops_cost = billable_iops * instance_months * iops_cost_per_iops

    # Throughput cost
    billable_throughput_mbps = max(0, provisioned_throughput_mbps - baseline_throughput_mbps)
    throughput_cost = billable_throughput_mbps * instance_months * throughput_cost_per_mbps

    # Initial snapshot cost
    initial_snapshot_cost = volume_size_gb * snapshot_cost_per_gb

    # Monthly snapshot cost
    monthly_snapshot_cost_per_snapshot = snapshot_size_gb * snapshot_cost_per_gb * 0.5  # Assuming 50% discount for partial storage month
    total_monthly_snapshot_cost = monthly_snapshot_cost_per_snapshot * snapshot_count

    # Total snapshot cost
    total_snapshot_cost = initial_snapshot_cost + total_monthly_snapshot_cost

    # Total EBS cost
    total_ebs_cost = storage_cost + iops_cost + throughput_cost + total_snapshot_cost

    data = {
        "EBS Storage Cost": storage_cost,
        "EBS IOPS Cost": iops_cost,
        "EBS Throughput Cost": throughput_cost,
        "EBS Snapshot Cost": total_snapshot_cost,
        "Total EBS Cost": total_ebs_cost
    }
    return total_ebs_cost

def calculate_gp3_volume_cost(volume_size_gb, provisioned_iops, baseline_iops=3000, storage_cost_per_gb=0.08, iops_cost_per_iops=0.005):
    # Calculate storage cost
    storage_cost = volume_size_gb * storage_cost_per_gb

    # Calculate provisioned IOPS cost
    if provisioned_iops > baseline_iops:
        iops_cost = (provisioned_iops - baseline_iops) * iops_cost_per_iops
    else:
        iops_cost = 0

    # Calculate total cost
    total_cost = storage_cost + iops_cost

    return total_cost

def calculate_cost_savings(severless_cost, aws_redis_cost):
    # Calculate the absolute savings
    savings = aws_redis_cost - severless_cost

    # Calculate the percentage savings
    percentage_savings = (savings / aws_redis_cost) * 100

    return percentage_savings

def impact_of_differnt_batch_sizes():
    dataset = 'imagenet'
    batch_sizes = [4,8,16,32,64,128,256,512]
    duration_hours = 730
    cache_proxy = 'c5n.xlarge'

    results = []

    #for percentage in range(10, 110, 10):
    for batch_size in  batch_sizes:

        num_files = dataset_info[dataset]['num_files']
        data_size_gb = dataset_info[dataset]['size_gb']

        estimated_severless_cost = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=batch_size,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )

        results.append(
            {'batch_size': batch_size,
             "cost": estimated_severless_cost})
    
    return results

def system_comaprsion_inc_batch_sizes():
    dataset = 'imagenet'
    batch_size = 128
    duration_hours = 730
    cache_proxy = 'c5n.xlarge'

    results = []

    #for percentage in range(10, 110, 10):
    for percentage in [15, 25, 40, 55, 70, 85, 100]:

        num_files = int(dataset_info[dataset]['num_files'] * (percentage / 100))
        data_size_gb = dataset_info[dataset]['size_gb'] * (percentage / 100)
        
        estimated_severless_cost_8_batch_size = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=8,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )
        estimated_severless_cost_64_batch_size = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=64,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )
        
        estimated_severless_cost_128_batch_size = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=128,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )
        
        estimated_severless_cost_256_batch_size = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=256,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )

     
        estimated_aws_redis_cost = caclaute_aws_redis_cache_cost(
            num_files=num_files,
            duration_hours=duration_hours,
            data_size_gb=data_size_gb
        )
       
        result = {
        
            "percentage": percentage,
            "aws_redis_cost": estimated_aws_redis_cost,
            # "elasticache_serverless_cost": estimated_elasticache_serverless,
            "severless_cost_8_batch_size": estimated_severless_cost_8_batch_size,
            "severless_cost_64_batch_size": estimated_severless_cost_64_batch_size,
            "severless_cost_128_batch_size": estimated_severless_cost_128_batch_size,
            # "severless_cost_256_batch_size": estimated_severless_cost_256_batch_size,
         }

        results.append(result)
    return results

# def system_comaprsion():
#     dataset = 'imagenet'
#     dataset_num_files = dataset_info[dataset]['num_files']
#     dataset_size_gb = dataset_info[dataset]['size_gb']
#     batch_size_files = 128
#     batch_size_gb = dataset_size_gb //batch_size_files
#     duration_hours = 730 #one month
#     cache_proxy = 'c5n.xlarge'
#     results = []
#     #for percentage in range(10, 110, 10):
#     # for percentage in [15, 25, 40, 55, 70, 85, 100]:
#     # for percentage in [100]:
#     for lookahead_distance in [50, 100, 200,400, 600, 800,1000]:
        
#         num_files = int(dataset_info[dataset]['num_files'] * (percentage / 100))
#         data_size_gb = dataset_info[dataset]['size_gb'] * (percentage / 100)

#         estimated_severless_cost = cacluate_serverless_cache_costs(
#             num_files=num_files,
#             batch_size=batch_size,
#             duration_hours=duration_hours,
#             ec2_instance_type=cache_proxy
#         )
#         estimated_aws_redis_cost = caclaute_aws_redis_cache_cost(
#             num_files=num_files,
#             duration_hours=duration_hours,
#             data_size_gb=data_size_gb
#         )
#         estimated_elasticache_serverless = caclaute_aws_elasticache_serverless_cache_cost(
#             data_size_gb=data_size_gb,
#             duration_hours=duration_hours
#         )
#         estimated_gp3_volume_cost = calculate_ebs_cost(
#             volume_size_gb=data_size_gb,
#             provisioned_iops=3000,
#             provisioned_throughput_mbps=125,
#             snapshot_size_gb=3,
#             snapshot_count=59.83
#         )

#         result = {
#             "percentage": percentage,
#             "aws_redis_cost": estimated_aws_redis_cost,
#             "elasticache_serverless_cost": estimated_elasticache_serverless,
#             "severless_cost": estimated_severless_cost,
#             }

#         results.append(result)
#     return results


def system_comaprsion():

    dataset = 'imagenet'
    dataset_num_files = dataset_info[dataset]['num_files']
    dataset_size_gb = dataset_info[dataset]['size_gb']
    batch_size_files = 128
    batch_size_gb = dataset_size_gb /(dataset_num_files//batch_size_files)
    duration_hours = 730 #one month
    cache_proxy = 'c5n.xlarge'
    results = []
    #for percentage in range(10, 110, 10):
    # for percentage in [15, 25, 40, 55, 70, 85, 100]:
    # for percentage in [100]:
    for minibatch_lookahead_distance in [1000,2500,5000,7500,10000,12500]:
        num_files = minibatch_lookahead_distance * batch_size_files
        # num_files = int(dataset_info[dataset]['num_files'] * (percentage / 100))
        # data_size_gb = dataset_info[dataset]['size_gb'] * (percentage / 100)
        data_size_gb = minibatch_lookahead_distance * batch_size_gb

        estimated_severless_cost = cacluate_serverless_cache_costs(
            num_files=num_files,
            batch_size=batch_size_files,
            duration_hours=duration_hours,
            ec2_instance_type=cache_proxy
        )
        estimated_aws_redis_cost = caclaute_aws_redis_cache_cost(
            num_files=num_files,
            duration_hours=duration_hours,
            data_size_gb=data_size_gb
        )
        estimated_elasticache_serverless = caclaute_aws_elasticache_serverless_cache_cost(
            data_size_gb=data_size_gb,
            duration_hours=duration_hours
        )
        estimated_gp3_volume_cost = calculate_ebs_cost(
            volume_size_gb=data_size_gb,
            provisioned_iops=3000,
            provisioned_throughput_mbps=125,
            snapshot_size_gb=3,
            snapshot_count=59.83
        )

        result = {
            "percentage": minibatch_lookahead_distance,
            "aws_redis_cost": estimated_aws_redis_cost,
            "elasticache_serverless_cost": estimated_elasticache_serverless,
            "severless_cost": estimated_severless_cost,
            "ebs_cost": estimated_gp3_volume_cost,

            }

        results.append(result)
        print(results)
    return results

def cost_of_training_jobs_serverless(num_jobs, num_epochs, num_files, batch_size, keep_alive_interval_mins = 15, ec2_instance_type ='c5n.2xlarge'):
    
    num_batches = num_files // batch_size

    #cacluate initial cost of populating the cache. Invoke fucntion that loads a batch worth of data from S3 and adds it to the cache
    create_batches_lmabda_cost = calculate_lambda_cost(invocation_count=num_batches, execution_duration_ms=3500, memory_allocation_mb=2048)
    s3_read_requests_costs = calculate_s3_read_request_cost(num_files)
    create_batches_lmabda_cost = (create_batches_lmabda_cost + s3_read_requests_costs) * num_epochs
    num_requests = (num_batches * num_jobs) * num_jobs
    keep_alive_cost = calculate_lambda_cost(invocation_count=num_requests, execution_duration_ms=500, memory_allocation_mb=1048)
    #cost of running ec2 instnace 
    proxy_cost = ec2_on_demand_cost_per_hour[ec2_instance_type] * duration_hours
    total_cost = create_batches_lmabda_cost + keep_alive_cost + proxy_cost
    
    return total_cost






if __name__ == "__main__":
    system_comaprsion()