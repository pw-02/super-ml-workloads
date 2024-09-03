import redis

def fetch_from_cache(batch_id, cache_host = "34.220.21.111", cache_port = 6378):
     try:
        cache_client =redis.StrictRedis = redis.StrictRedis(host=cache_host, port=cache_port)
        return cache_client.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
        

if __name__ == "__main__":
   respsonse = fetch_from_cache("1")
   
   print(respsonse)