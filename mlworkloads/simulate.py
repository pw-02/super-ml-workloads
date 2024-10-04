from dataloading.super.super_mapped_dataset import SUPERMappedDataset
from dataloading.super.super_sampler import SUPERSampler
from torch.utils.data import DataLoader
import time
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DL_TIME_FOR_CACHE_MISS = 1.4
DL_TIME_FOR_CACHE_HIT = 0.001
GPU_TIME = 0.01
NUM_JOBS = 1 # Number of parallel jobs to simulate
NUM_PYTORCH_WORKERS = 4 # Number of workers for PyTorch DataLoader
DELAY_BETWEEN_JOBS_START = 2  # Delay in seconds between the start of each job
MAX_BATCHES_PER_JOB = 10000  # Number of batches each job will process
MAX_EPOCHS = 3  # Number of epochs each job will run
LOG_ITERATION = 20  # Log every n iterations


def simulate_training_job(job_id):
    #prepare the dataloader
    dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/train/", 
                                 transform=None, 
                                 cache_address=None,
                                 simulate_mode=True,
                                 simulate_time_for_cache_miss=DL_TIME_FOR_CACHE_MISS,
                                 simulate_time_for_cache_hit=DL_TIME_FOR_CACHE_HIT)
    sampler = SUPERSampler(dataset, "localhost:50051")
    train_dataloader:DataLoader = DataLoader(dataset, sampler=sampler, num_workers=NUM_PYTORCH_WORKERS, batch_size=None)  # batch_size=None since sampler provides batches
    cache_hit_counter = 0
    cache_miss_counter = 0
    job_started = time.perf_counter()
    global_batch_idx = 0

    for epoch in range(MAX_EPOCHS):
        end = time.perf_counter()
        for  batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(train_dataloader):
            batch_id = batch
            
            wait_for_data_time = time.perf_counter() - end
            
            if global_batch_idx + 1 == MAX_BATCHES_PER_JOB:
                if isinstance(train_dataloader.sampler, SUPERSampler):
                    train_dataloader.sampler.send_job_ended_notfication()
                break
            if is_cache_hit:
                cache_hit_counter += 1
            else:
                cache_miss_counter += 1

            global_batch_idx+=1

            time.sleep(GPU_TIME)

            if batch_idx % LOG_ITERATION == 0:
                logger.info(f"Job {job_id}, Setep:{batch_idx+1}, Batch: {batch_id} Cache Hits: {cache_hit_counter}, Cache Misses:{cache_miss_counter}, IterTime: {time.perf_counter() - end:.4f}")


            if isinstance(train_dataloader.sampler, SUPERSampler):
                train_dataloader.sampler.send_job_update_to_super(
                        batch_id,
                        wait_for_data_time,
                        is_cache_hit,
                        GPU_TIME,
                        cached_on_miss
                    )
            end = time.perf_counter()
        
        if global_batch_idx + 1 == MAX_BATCHES_PER_JOB or epoch + 1 == MAX_EPOCHS:
            break
    
    job_duration = time.perf_counter() - job_started
    bacthes_per_seond = global_batch_idx / job_duration
    return job_id, job_duration, global_batch_idx, bacthes_per_seond, cache_hit_counter, cache_miss_counter, cache_hit_counter / (cache_hit_counter + cache_miss_counter) if (cache_hit_counter + cache_miss_counter) > 0 else 0

if __name__ == "__main__":
    job_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_JOBS) as executor:
        futures = [] 
        for job_id in range(1, NUM_JOBS + 1):
            # Submit job with a delay
            logger.info(f"Starting job {job_id} after {DELAY_BETWEEN_JOBS_START * (job_id - 1)} seconds delay.")
            future = executor.submit(simulate_training_job, str(job_id))
            futures.append(future)
            # Delay the start of the next job
            time.sleep(DELAY_BETWEEN_JOBS_START)

        # Collect and log the results of each job
        for future in concurrent.futures.as_completed(futures):
            job_id, job_duration, batches_processed, bacthes_per_seond, cache_hits, cache_misses, hit_rate = future.result()            
            job_results.append((job_id, job_duration, batches_processed, bacthes_per_seond, cache_hits, cache_misses, hit_rate))
    
    total_duration = sum(result[1] for result in job_results)
    total_bataches = sum(result[2] for result in job_results)
    total_hits = sum(result[4] for result in job_results)
    total_misses = sum(result[5] for result in job_results)
    total_ratio = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    toatl_bacthes_per_seond = total_duration / total_bataches

    for result in job_results:
        job_id, job_duration, batches_processed, bacthes_per_seond, cache_hits, cache_misses, hit_rate = result
        logger.info(f"Results for Job {job_id} - duration:{job_duration:.3f}, batches:{batches_processed}, speed:{bacthes_per_seond:.3f}, Hits:{cache_hits}, Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}")          

    logger.info(f"Total Results: duration:{total_duration:.3f}, batches:{total_bataches}, speed:{toatl_bacthes_per_seond:.3f}, Hits:{total_hits}, Misses:{total_misses}, Hit Rate: {total_ratio:.2f}")

    # logger.info(f"Total duration={total_duration}, total_batches/s={toatl_bacthes_per_seond}, Total Cache Hits={total_hits}, Total Cache Misses={total_misses}, Total Hit Rate ={total_ratio:.2f}")
    time.sleep(5)
