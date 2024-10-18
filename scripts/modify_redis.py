import redis
import random

# Connect to Redis
redis_client = redis.Redis(host='coco.rdior4.ng.0001.usw2.cache.amazonaws.com:6379', port=6379, db=0)  # Adjust parameters as needed

# Step 1: Get all keys in the cache
all_keys = redis_client.keys('*')

# Step 2: Calculate how many keys to remove (25%)
num_to_remove = int(len(all_keys) * 0.25)

# Step 3: Randomly select keys to delete
keys_to_remove = random.sample(all_keys, num_to_remove)

# Step 4: Remove selected keys from Redis
if keys_to_remove:
    redis_client.delete(*keys_to_remove)  # Batch delete

print(f'Removed {len(keys_to_remove)} keys from the Redis cache.')
