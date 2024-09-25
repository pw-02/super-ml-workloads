import redis
import lz4.frame
from io import BytesIO
import torch


cache_client = redis.StrictRedis(host="44.243.9.209", port=6378)
cache_client.set("batch_id", "hello")
print(cache_client.get("batch_id"))
def fetch_from_cache(batch_id):
     try:
        return cache_client.get(batch_id)
     except Exception as e:
        print(f"Error fetching from cache: {e}")
        return None
def _torch_batch_to_bytes(self, images, text,text_atts,ids) -> str:
        with BytesIO() as buffer:
            torch.save((images, text,text_atts,ids), buffer)
            bytes_minibatch = buffer.getvalue()
            bytes_minibatch = lz4.frame.compress(bytes_minibatch)
        return bytes_minibatch         

def _bytes_to_torch_batch(bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with BytesIO(compressed_batch) as buffer:
            images, text,text_atts,ids  = torch.load(buffer)
        return images, text,text_atts,ids 

if __name__ == "__main__":
      data = fetch_from_cache('1_1_5_a459f276d19e8810')
      images, text,text_atts,ids  = _bytes_to_torch_batch(data)
      pass

