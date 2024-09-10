import grpc
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
from torch.utils.data import Sampler
import hashlib
import uuid
import time

class SUPERSampler(Sampler):
    def __init__(self, dataset, grpc_server_address, batch_size=32):
        # self.job_id = str(os.getpid())  # Unique job ID for the current process
        self.job_id = str(uuid.uuid4())  # Unique job ID for the current process
        self.batch_size = batch_size
        self.dataset = dataset
        self.grpc_server_address = grpc_server_address
        self.total_batches = None
        self.stub = self._create_grpc_stub()
        self._register_dataset_with_super()
        self.current_batch = 0
        self.previous_step_is_cache_hit = None
        self.previous_step_wait_for_data_time = None
        self.previous_step_gpu_time = None
        self.cached_previous_batch = False
        self.previous_step_idx = None


    def _test_grpc_connection(self):
        try:
            # Example ping to test connection
            self.stub.Ping(minibatch_service_pb2.PingRequest(message='ping'))
            print("Connection to SUPER server confirmed. Registering as a client...")
        except grpc.RpcError as e:
            print(f"Failed to connect to SUPER server: {e.details()}")
            exit(1)

    def _create_grpc_stub(self):
        channel = grpc.insecure_channel(self.grpc_server_address)
        stub = minibatch_service_pb2_grpc.MiniBatchServiceStub(channel)
        return stub
    
    def _register_dataset_with_super(self):
        try:
            response = self.stub.RegisterDataset(minibatch_service_pb2.RegisterDatasetRequest(
                data_dir=self.dataset.s3_data_dir))
           
            print(f"{response.message}")
            self.total_batches = response.total_batches
        except grpc.RpcError as e:
            print(f"Failed to register dataset: {e.details()}")
            exit(1)
    
    def _gen_batch_id(self, indicies):
    # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in indicies)
        # Hash the concatenated string to generate a unique ID
        unique_id = hashlib.sha1(id_string.encode()).hexdigest() 
        return unique_id
    
    def set_step_perf_metrics(self, step_idx, previous_step_wait_for_data_time: float, previous_step_is_cache_hit: bool, previous_step_gpu_time: float, cached_previous_batch: bool):
        self.previous_step_wait_for_data_time = previous_step_wait_for_data_time
        self.previous_step_is_cache_hit = previous_step_is_cache_hit
        self.previous_step_gpu_time = previous_step_gpu_time
        self.cached_previous_batch = cached_previous_batch
        self.previous_step_idx = step_idx
        
    # def reset_step_perf_metrics(self):
    #     self.previous_step_wait_for_data_time = None
    #     self.previous_step_is_cache_hit = None
    #     self.previous_step_gpu_time = None
    #     self.cached_previous_batch = False
    
    def send_job_ended_notfication(self):
        try:
            self.stub.JobEnded(minibatch_service_pb2.JobEndedRequest(
                job_id=self.job_id, 
                data_dir=self.dataset.s3_data_dir,
                previous_step_time = self.previous_step_wait_for_data_time,
                previous_step_is_cache_hit = self.previous_step_is_cache_hit,
                cached_previous_batch = self.cached_previous_batch
                ))
        except grpc.RpcError as e:
            print(f"Failed to send job ended notification: {e.details()}")

    def __iter__(self):
        while True:
            for _ in range(self.total_batches):
                try:  
                    response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
                        job_id=self.job_id,
                        data_dir=self.dataset.s3_data_dir,
                        previous_step_idx = self.previous_step_idx,
                        previous_step_wait_for_data_time = self.previous_step_wait_for_data_time,
                        previous_step_is_cache_hit = self.previous_step_is_cache_hit,
                        previous_step_gpu_time = self.previous_step_gpu_time,
                        cached_previous_batch = self.cached_previous_batch))
                    
                    batch_id = response.batch.batch_id
                    batch_indices = list(response.batch.indicies)
                    is_cached = response.batch.is_cached
        
                    self.current_batch += 1
                    yield batch_id, batch_indices, is_cached

                except grpc.RpcError as e:
                    print(f"Failed to fetch batch: {e.details()}")
                    continue
            
            # Reset the batch counter at the end of each epoch
            self.current_batch = 0
            break  # Exit the loop after completing one epoch

    def __len__(self):
        return self.total_batches

# if __name__ == "__main__":
#     import torchvision.transforms as transforms
#     # Example usage
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
#     sampler = SUPERSampler(dataset, "localhost:50051")

#     for batch_idx, (batch_id, batch_indices) in enumerate(sampler):
#         pass

#     # Example DataLoader usage with custom sampler
#     from torch.utils.data import DataLoader
#     for num_worker in [4]:
#         dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
#         sampler = SUPERSampler(dataset, "localhost:50051")
#         print(f"Number of workers: {num_worker}")
#         total_fetch_time = 0
#         total_transform_time = 0
#         total_dataloading_delay = 0
#         dataloader = DataLoader(dataset, sampler=sampler, num_workers=num_worker, batch_size=None)  # batch_size=None since sampler provides batches
#         total_steps = 10
#         num_epochs = 1
#         step_count = 0
#         for epoch in range(num_epochs):
#             # print(f"Epoch {epoch + 1}")
#             end = time.perf_counter()
#             for batch_idx, (batch_id, data_fetch_time, transformation_time, cache_hit) in enumerate(dataloader):
#                 delay = time.perf_counter() - end
#                 total_dataloading_delay += delay
#                 step_count += 1
#                 total_fetch_time += data_fetch_time
#                 total_transform_time += transformation_time
#                 print(f"Batch index: {batch_idx + 1}, Batch_id {batch_id}, Fetch Time: {data_fetch_time}, Transfrom Time: {transformation_time}, Dataloading Delay: {delay}")
#                 end = time.perf_counter()
#                 if step_count >= total_steps:
#                     break
#         print(f"Numer Workers: {num_worker},  Total fetch time: {total_fetch_time}, Total transform time: {total_transform_time}, total_dataloading_delay: {total_dataloading_delay}")
#         print("---------------------------------------------------")