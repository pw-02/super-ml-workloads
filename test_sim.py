import time
from threading import Thread, Lock
from collections import deque, defaultdict

class PredictiveBatchCache:
    def __init__(self, load_time, cache_size, ttl):
        self.load_time = load_time
        self.cache_size = cache_size
        self.ttl = ttl
        self.cache = deque()  # Store batches as (batch_id, last_accessed, accessed_by_jobs)
        self.access_times = {}  # Batch ID -> Last accessed time
        self.batch_job_access = defaultdict(set)  # Batch ID -> Set of job IDs that have accessed it
        self.job_access_times = defaultdict(lambda: (0, 0))  # Job ID -> (processing_rate, start_time)
        self.lock = Lock()
        self.preloading_threads = []

    def add_job(self, job_id, processing_rate, start_time):
        """
        Adds a job to the cache system and schedules preloading for its batches.
        """
        self.job_access_times[job_id] = (processing_rate, start_time)
        self.schedule_preloading(job_id, processing_rate, start_time)

    def calculate_preload_time(self, batch_id, job_id):
        """
        Calculates the time when the batch should start being preloaded to be ready in time.
        """
        processing_rate, start_time = self.job_access_times[job_id]
        access_time = start_time + batch_id / processing_rate
        return access_time - self.load_time

    def preload_batch(self, batch_id):
        """
        Simulates loading a batch into the cache.
        """
        time.sleep(self.load_time)
        with self.lock:
            # Remove old batches if necessary
            self.cleanup_cache()
            # Add the new batch to the cache
            current_time = time.time()
            if batch_id in self.access_times:
                # Update existing batch
                self.access_times[batch_id] = current_time
                for i, (cached_id, _, _) in enumerate(self.cache):
                    if cached_id == batch_id:
                        self.cache[i] = (batch_id, current_time, self.batch_job_access[batch_id])
                        break
            else:
                # Add new batch
                self.cache.append((batch_id, current_time, set()))
                self.access_times[batch_id] = current_time
            print(f"Batch {batch_id} loaded into cache.")

    def schedule_preloading(self, job_id, processing_rate, start_time):
        """
        Schedules preloading for all batches required by a job.
        """
        self.job_access_times[job_id] = (processing_rate, start_time)
        # Determine the range of batches to preload (adjust as needed)
        for batch_id in range(20):  # Example: Preload the first 20 batches
            preload_time = self.calculate_preload_time(batch_id, job_id)
            delay = preload_time - time.time()
            if delay > 0:
                thread = Thread(target=self.delayed_preload, args=(batch_id, delay))
                self.preloading_threads.append(thread)
                thread.start()

    def delayed_preload(self, batch_id, delay):
        """
        Delays the preload of a batch based on the calculated time.
        """
        time.sleep(delay)
        self.preload_batch(batch_id)

    def get_batch(self, batch_id, job_id):
        """
        Retrieves a batch from the cache and updates its access time.
        """
        with self.lock:
            for i, (cached_id, last_accessed, accessed_by_jobs) in enumerate(self.cache):
                if cached_id == batch_id:
                    current_time = time.time()
                    self.access_times[batch_id] = current_time
                    self.cache[i] = (cached_id, current_time, accessed_by_jobs | {job_id})
                    self.batch_job_access[batch_id].add(job_id)
                    return batch_id
            return None

    def cleanup_cache(self):
        """
        Cleans up the cache by removing batches that have expired based on TTL and have been accessed by all jobs.
        """
        current_time = time.time()
        with self.lock:
            to_keep = []
            for batch_id, last_accessed, accessed_by_jobs in self.cache:
                if (current_time - last_accessed <= self.ttl and
                    len(self.batch_job_access[batch_id]) < len(self.job_access_times)):
                    # Keep batches that haven't been accessed by all jobs yet
                    to_keep.append((batch_id, last_accessed, accessed_by_jobs))
                elif current_time - last_accessed <= self.ttl:
                    # Keep batches that have been accessed by all jobs and are within TTL
                    to_keep.append((batch_id, last_accessed, accessed_by_jobs))
            
            self.cache = deque(to_keep)
            self.access_times = {batch_id: last_accessed for batch_id, last_accessed, _ in to_keep}

# Example Usage
load_time = 5  # Time to load each batch in seconds
cache_size = 10  # Number of batches to keep in cache (not used in this case, since size is unlimited)
ttl = 15 * 60  # Time-to-Live for each batch in seconds (15 minutes)
predictive_cache = PredictiveBatchCache(load_time, cache_size, ttl)

# Add multiple jobs with different processing rates and start times
jobs = [
    (1, 2, time.time() + 10),  # Job 1: processes 2 batches per second, starts in 10 seconds
    (2, 1, time.time() + 15),  # Job 2: processes 1 batch per second, starts in 15 seconds
]

for job_id, processing_rate, start_time in jobs:
    predictive_cache.add_job(job_id, processing_rate, start_time)

# Simulate batch retrieval
time.sleep(12)  # Wait for some time to allow preloading
print(predictive_cache.get_batch(0, 1))  # Should print 0 if batch 0 is preloaded by job 1
print(predictive_cache.get_batch(0, 2))  # Should print 0 if batch 0 is preloaded by job 2

# To simulate TTL expiry, you could wait further and observe cleanup
time.sleep(ttl + 10)  # Wait for TTL to expire
predictive_cache.cleanup_cache()  # Cleanup expired batches
