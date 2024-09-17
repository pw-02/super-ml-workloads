from dataloading.super.super_mapped_dataset import SUPERMappedDataset
from dataloading.super.super_sampler import SUPERSampler
from torch.utils.data import DataLoader
import time

if __name__ == "__main__":
    import torchvision.transforms as transforms
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", 
                                 transform=transform, 
                                 cache_address=None,
                                 simulate_mode=True,
                                 simulate_time_for_cache_miss=0.2,
                                 simulate_time_for_cache_hit=0.02)
    sampler = SUPERSampler(dataset, "localhost:50051")
    dataloader:DataLoader = DataLoader(dataset, sampler=sampler, num_workers=0, batch_size=None)  # batch_size=None since sampler provides batches
    cache_hits = 0
    cache_misses = 0
    previous_step_training_time = 0
    previous_step_is_cache_hit = False
    TIME_ON_CACHE_HIT = 0.045
    TIME_ON_CACHE_MISS = 0.25
    GPU_TIME=0.1
    end = time.perf_counter()

    for  batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(dataloader):
        if is_cache_hit:
            previous_step_is_cache_hit = True
            cache_hits += 1
            time.sleep(TIME_ON_CACHE_HIT)
            previous_step_training_time =TIME_ON_CACHE_HIT
        else:
            cache_misses += 1
            previous_step_is_cache_hit = False
            time.sleep(TIME_ON_CACHE_MISS)
            previous_step_training_time =TIME_ON_CACHE_MISS

        hit_rate = cache_hits / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
        print(f'Batch {batch_idx+1}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}, {time.perf_counter() - end}')
        if isinstance(sampler, SUPERSampler):
            sampler.set_step_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, GPU_TIME, False)
    
    if isinstance(sampler, SUPERSampler):
        sampler.send_job_ended_notfication()
    print(f"Job ended. Total duration: {time.perf_counter() - end}")



    # for batch_idx, (batch_id, data_fetch_time, transformation_time, is_cache_hit) in enumerate(dataloader):
    #     if is_cache_hit:
    #         previous_step_is_cache_hit = True
    #         cache_hits += 1
    #         time.sleep(TIME_ON_CACHE_HIT)
    #         previous_step_training_time =TIME_ON_CACHE_HIT
    #     else:
    #         cache_misses += 1
    #         previous_step_is_cache_hit = False
    #         time.sleep(TIME_ON_CACHE_MISS)
    #         previous_step_training_time =TIME_ON_CACHE_MISS

    #     hit_rate = cache_hits / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
    #     print(f'Batch {batch_idx+1}: {batch_id}, Cache Hits: {cache_hits}, Cache Misses:{cache_misses}, Hit Rate: {hit_rate:.2f}, {time.perf_counter() - end}')
    #     if isinstance( dataloader.sampler, SUPERSampler):
    #         dataloader.sampler.set_step_perf_metrics(previous_step_training_time, previous_step_is_cache_hit, GPU_TIME)
    
    # if isinstance( dataloader.sampler, SUPERSampler):
    #     dataloader.sampler.send_job_ended_notfication()

    # print(f"Job ended. Total duration: {time.perf_counter() - end}")

    # for batch_idx, (batch_id, batch_indices,is_cached) in enumerate(sampler):
    #     sampler.set_step_perf_metrics(0.1, True)
    #     pass
