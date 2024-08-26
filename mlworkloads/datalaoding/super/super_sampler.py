import grpc
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
from torch.utils.data import Sampler
from datalaoding.s3.s3_mapped_dataset import S3MappedDataset
import os
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

    def __iter__(self):
        while True:
            for _ in range(self.total_batches):
                try:  
                    response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
                        job_id=self.job_id,
                        data_dir=self.dataset.s3_data_dir))
                    
                    batch_id = response.batch.batch_id
                    batch_indices = list(response.batch.indicies)
                    is_cached = response.batch.is_cached
                    # # Simulate fetching a batch from the service
                    # batch_indices = list(range(self.batch_size))  # Dummy indices
                    # batch_id = self._gen_batch_id(batch_indices)

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

if __name__ == "__main__":
    import torchvision.transforms as transforms
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # for batch_idx, (batch_id, batch_indices) in enumerate(sampler):
    #     pass

    # Example DataLoader usage with custom sampler
    from torch.utils.data import DataLoader
    for num_worker in [4]:
        dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
        sampler = SUPERSampler(dataset, "localhost:50051")
        print(f"Number of workers: {num_worker}")
        total_fetch_time = 0
        total_transform_time = 0
        total_dataloading_delay = 0
        dataloader = DataLoader(dataset, sampler=sampler, num_workers=num_worker, batch_size=None)  # batch_size=None since sampler provides batches
        total_steps = 10
        num_epochs = 1
        step_count = 0
        for epoch in range(num_epochs):
            # print(f"Epoch {epoch + 1}")
            end = time.perf_counter()
            for batch_idx, (batch_id, data_fetch_time, transformation_time, cache_hit) in enumerate(dataloader):
                delay = time.perf_counter() - end
                total_dataloading_delay += delay
                step_count += 1
                total_fetch_time += data_fetch_time
                total_transform_time += transformation_time
                print(f"Batch index: {batch_idx + 1}, Batch_id {batch_id}, Fetch Time: {data_fetch_time}, Transfrom Time: {transformation_time}, Dataloading Delay: {delay}")
                end = time.perf_counter()
                if step_count >= total_steps:
                    break
        print(f"Numer Workers: {num_worker},  Total fetch time: {total_fetch_time}, Total transform time: {total_transform_time}, total_dataloading_delay: {total_dataloading_delay}")
        print("---------------------------------------------------")