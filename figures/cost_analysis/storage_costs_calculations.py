from figures.cost_analysis.constants import *

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
                                                       execution_duration_ms=500,
                                                       memory_allocation_mb=1048)
    #cost of running ec2 instnace 
    proxy_cost = ec2_on_demand_cost_per_hour[ec2_instance_type] * duration_hours
    total_cost = create_batches_lmabda_cost + keep_alive_cost + proxy_cost
    return total_cost

def caclaute_aws_redis_cache_cost(num_files, duration_hours, redis_instacne ='cache.m5.12xlarge'):
    cache_cost = elasticache_redis_prices_per_hour[redis_instacne] * duration_hours
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
def main():
    dataset = 'imagenet'
    batch_size = 128
    duration_hours = 730
    cache_proxy = 'c5n.xlarge'
    estimated_severless_cost = cacluate_serverless_cache_costs (num_files = dataset_info[dataset]['num_files'], batch_size = batch_size, duration_hours = duration_hours, ec2_instance_type=cache_proxy )
    estimated_aws_redis_cost = caclaute_aws_redis_cache_cost(num_files= dataset_info[dataset]['num_files'], duration_hours=duration_hours,redis_instacne='cache.m5.12xlarge')
    estimated_elasticache_serverless = caclaute_aws_elasticache_serverless_cache_cost(
        data_size_gb=dataset_info[dataset]['size_gb'],
        duration_hours= duration_hours
    )
    estimated_gp3_volume_cost = calculate_ebs_cost(
        volume_size_gb=dataset_info[dataset]['size_gb'],
        provisioned_iops=3000,
        provisioned_throughput_mbps=125,
        snapshot_size_gb=3,
        snapshot_count = 59.83
    )

    print(f"Estimated Severless cost: ${estimated_severless_cost:.8f}")
    print(f"Estimated EasltiCache cost: ${estimated_aws_redis_cost:.8f}")
    print(f"Estimated EasltiCache Serverless cost: ${estimated_elasticache_serverless:.8f}")
    print(f"Estimated EBS cost: ${estimated_gp3_volume_cost:.8f}")

    print(f"Cost savings v EasltiCache Redis: {calculate_cost_savings(estimated_severless_cost, estimated_aws_redis_cost):.2f}%")
    print(f"Cost savings v EasltiCache Serverless: {calculate_cost_savings(estimated_severless_cost, estimated_elasticache_serverless):.2f}%")
    print(f"Cost losses v EBS: {calculate_cost_savings(estimated_gp3_volume_cost,estimated_severless_cost ):.2f}%")

   

if __name__ == "__main__":
    main()

# print(f"Estimated Lambda cost: ${estimated_cost:.2f}")


