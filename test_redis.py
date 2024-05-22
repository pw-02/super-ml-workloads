import redis
cache_host, cache_port = '10.0.24.192:6378'.split(":")
cache_client = redis.StrictRedis(host=cache_host, port=cache_port)
cache_client.set('foo', 'bar')
resposne = cache_client.get('foo')
print(resposne)