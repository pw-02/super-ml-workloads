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
elasticache_redis_prices_per_hour = {
    'cache.t2.micro': 0.017,
    'cache.t3.micro': 0.017,
    'cache.t3.small': 0.034,
    'cache.t3.medium': 0.067,
    'cache.t3.large': 0.134,
    'cache.m5.12xlarge': 3.744 
}

elasticache_serverless_prices_per_gb_hour =  0.125

dataset_info = {
    'imagenet': {"num_files": 1096302, 'size_gb': 122.3}
    }