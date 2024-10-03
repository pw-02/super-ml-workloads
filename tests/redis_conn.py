import redis



cache_client =redis.StrictRedis = redis.StrictRedis(host="54.202.145.114",
                                                     port=6378,
                                              
                                                )

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
      while True:
          for i in range(1000):
            batch_id = 'batch_id{}'.format(2)
            put_in_cache(batch_id)
            print(fetch_from_cache(batch_id))