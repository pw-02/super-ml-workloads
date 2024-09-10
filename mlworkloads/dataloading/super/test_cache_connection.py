import redis
name: super
# grpc_server_address: 44.243.2.122:50051
# cache_address: 44.243.2.122:6378 #null
def fetch_from_cache(batch_id, cache_host = "44.243.2.122", cache_port = 6378):
     try:
        cache_client =redis.StrictRedis = redis.StrictRedis(host=cache_host, port=cache_port)
        return cache_client.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
        

if __name__ == "__main__":
   respsonse = fetch_from_cache("1")
   
   print(respsonse)