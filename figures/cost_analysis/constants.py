s3_storage_cost_per_gb = 0.023  # $0.023 per GB per month
s3_data_transfer_cost_per_gb = 0.09  # $0.09 per GB data transfer out to the internet
s3_get_request_cost = 0.0000004


ebs_storage_cost_per_gb_month = 0.10  # $0.10 per GB-month
ebs_data_transfer_cost_per_gb = 0.05  # $0.05 per GB data transfer out

ec2_on_demand_cost_per_hour = {
    't3.micro': 0.0116,    # $0.0116 per hour
    'm5.large': 0.096,     # $0.096 per hour
    # Add more instance types as needed
    'c5n.2xlarge': 0.432,  # $0.504 per hour (US East - N. Virginia)
    'c5n.xlarge': 0.216
}

def find_redis_node(gb_size):
    # Initialize variables to store the best match
    best_instance = None
    best_memory = float('inf')  # Start with infinity to find the minimum memory that is sufficient

    for instance in elasticache_redis_instances:
        memory_gib = instance['memory_gib']
        
        # Check if the instance's memory is greater than or equal to the required size
        if memory_gib >= gb_size:
            # Check if this instance has the smallest memory capacity that meets or exceeds the required size
            if memory_gib < best_memory:
                best_memory = memory_gib
                best_instance = instance

    return best_instance


elasticache_redis_instances = [
 {"instance_type": "cache.t2.micro", "memory_gib": 0.555, "cost_per_hour_usd": 0.018},
{"instance_type": "cache.t2.small", "memory_gib": 1.555, "cost_per_hour_usd": 0.036},
{"instance_type": "cache.t2.medium", "memory_gib": 3.22, "cost_per_hour_usd": 0.070},
{"instance_type": "cache.t3.micro", "memory_gib": 0.555, "cost_per_hour_usd": 0.015},
{"instance_type": "cache.t3.small", "memory_gib": 1.555, "cost_per_hour_usd": 0.034},
{"instance_type": "cache.t3.medium", "memory_gib": 3.22, "cost_per_hour_usd": 0.065},
{"instance_type": "cache.m5.large", "memory_gib": 6.42, "cost_per_hour_usd": 0.094},
{"instance_type": "cache.m5.xlarge", "memory_gib": 13.38, "cost_per_hour_usd": 0.188},
{"instance_type": "cache.m5.2xlarge", "memory_gib": 27.94, "cost_per_hour_usd": 0.376},
{"instance_type": "cache.m5.4xlarge", "memory_gib": 55.88, "cost_per_hour_usd": 0.752},
{"instance_type": "cache.m5.12xlarge", "memory_gib": 167.64, "cost_per_hour_usd": 2.256},
{"instance_type": "cache.m5.24xlarge", "memory_gib": 351.32, "cost_per_hour_usd": 4.512},
{"instance_type": "cache.r5.large", "memory_gib": 13.38, "cost_per_hour_usd": 0.132},
{"instance_type": "cache.r5.xlarge", "memory_gib": 27.94, "cost_per_hour_usd": 0.264},
{"instance_type": "cache.r5.2xlarge", "memory_gib": 55.88, "cost_per_hour_usd": 0.528},
{"instance_type": "cache.r5.4xlarge", "memory_gib": 111.76, "cost_per_hour_usd": 1.056},
{"instance_type": "cache.r5.12xlarge", "memory_gib": 335.28, "cost_per_hour_usd": 3.168},
{"instance_type": "cache.r5.24xlarge", "memory_gib": 669.56, "cost_per_hour_usd": 6.336},
{"instance_type": "cache.r6g.12xlarge", "memory_gib": 754.5, "cost_per_hour_usd": 3.888}

  ]


elasticache_serverless_prices_per_gb_hour =  0.125

dataset_info = {
    'imagenet': {"num_files": 10963020, 'size_gb': 122.3}
    }