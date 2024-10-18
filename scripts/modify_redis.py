import redis
import random

# Connect to Redis
redis_client = redis.Redis(host='coco.rdior4.ng.0001.usw2.cache.amazonaws.com', port=6379, db=0)  # Adjust parameters as needed

# Total number of keys in the cache
total_keys = 82688

# Step 1: Get all keys in the cache
all_keys = redis_client.keys('*')

# Ensure we do not exceed the total number of keys in the cache
current_key_count = len(all_keys)
print(f'Current number of keys in cache: {current_key_count}')

# Step 2: Calculate how many keys we need to keep (50%)
min_keys_to_keep = max(int(total_keys * 0.5), 1)  # Ensure at least 1 key remains
print(f'Minimum keys to keep in cache: {min_keys_to_keep}')

# Step 3: Calculate how many keys to remove
num_to_remove = max(0, current_key_count - min_keys_to_keep)  # Calculate keys to remove
print(f'Number of keys to remove: {num_to_remove}')

# Step 4: Randomly select keys to delete if there are keys to remove
if num_to_remove > 0:
    keys_to_remove = random.sample(all_keys, num_to_remove)  # Select random keys to delete

    # Step 5: Remove selected keys from Redis
    redis_client.delete(*keys_to_remove)  # Batch delete
    print(f'Removed {len(keys_to_remove)} keys from the Redis cache.')
else:
    print('No keys need to be removed; the cache is already within the desired limits.')
