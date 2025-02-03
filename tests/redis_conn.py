import redis


# cache_client =redis.StrictRedis = redis.StrictRedis(host="",
#                                                      port=6379,
#                                                      ssl=True)

cache_client =redis.StrictRedis = redis.StrictRedis(host="127.0.0.1",
                                                     port=6379)

def put_in_cache(batch_id):
     try:
        cache_client.set(batch_id, 'hello')
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None

def fetch_from_cache(batch_id):
     try:
        return cache_client.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
        

if __name__ == "__main__":
   batch_id = 'batch_id_{}'.format(1)
   put_in_cache(batch_id)
   print(fetch_from_cache(batch_id))