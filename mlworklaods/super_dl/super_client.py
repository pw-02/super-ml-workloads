import grpc
import mlworklaods.super_dl.cache_coordinator_pb2 as cache_coordinator_pb2
import mlworklaods.super_dl.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import logging
from mlworklaods.super_dl.cache_coordinator_pb2_grpc import CacheCoordinatorServiceStub

def configure_logger():
    # Set the log levels for specific loggers to WARNING
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

    # Configure the root logger with a custom log format
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("SUPERWorkload")
    return logger

logger = configure_logger()



class SuperClient:
    def __init__(self, super_addresss='localhost:50051'):
        self.super_addresss = super_addresss
        # Establish a connection to the gRPC server
        self.channel = grpc.insecure_channel(self.super_addresss)
        # Create a stub (client)
        self.stub:CacheCoordinatorServiceStub = cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(self.channel)
        
    def ping_server(self):
        try:
            ping_response = self.stub.Ping(cache_coordinator_pb2.PingRequest(message='ping'))
            logger.info(f"Ping Response: {ping_response.message}")
        except Exception as e:
            logger.error(f"Error in ping_server request: {e}")

    def register_job(self, job_id:int, data_dir:str):
        try:       
            register_job_response = self.stub.RegisterJob(cache_coordinator_pb2.RegisterJobRequest(
                job_id=job_id, 
                data_dir=data_dir))
            logger.info(f"Register job Response: {register_job_response.message}")
            return register_job_response.job_registered
        except Exception as e:
            logger.error(f"Error in register_job request: {e}")
    
    def get_next_batch(self, job_int:int, num_batches_requested:int = 1):
        try:   
            next_batch_response = self.stub.GetNextBatchToProcess(
                cache_coordinator_pb2.GetNextBatchRequest(
                    job_id=job_int, 
                    num_batches_requested=num_batches_requested))   
            
            # for batch in next_batch_response.batches:
                # logger.info(f"Received batch: {batch.batch_id}, {batch.indicies}, {batch.is_cached}")
            return next_batch_response.batches
        except Exception as e:
            logger.error(f"Error in get_next_batch request: {e}")
    
    def get_dataset_details(self, data_dir:str =''):
        try:   
            dataset_info_response = self.stub.GetDatasetInfo(
                cache_coordinator_pb2.DatasetInfoRequest(
                    data_dir=data_dir))   
            
            return dataset_info_response
        
        except Exception as e:
            logger.error(f"Error in get_dataset_details request: {e}")
    
    
    def job_ended_notification(self, job_int:int):
        try:   
            self.stub.JobEnded(cache_coordinator_pb2.JobEndedRequest(job_id=job_int))
            logger.info("Job Ended notification sent")

        except Exception as e:
            logger.error(f"Error in job_ended_notification request: {e}")

    def __del__(self):
        # Close the gRPC channel when the client is deleted
        self.channel.close()


def test_connections():
    import multiprocessing
    import time

    def worker(process_id, server_address):
        # Each worker creates its own instance of CacheCoordinatorClient
        client = SuperClient(server_address)
        while True:     
            # Example: Call GetBatchStatus RPC
            response = client.get_batch_status(process_id, 'example_dataset')
            print(f"Process {process_id}: GetBatchStatus Response - {response}")
            time.sleep(2)

    # Close the gRPC channel when the worker is done
    #del client
    
    # Example of using the CacheCoordinatorClient with multiple processes
    server_address = 'localhost:50051'
    num_processes = 4

    # Create a list to hold the process objects
    processes = []

    # Create a super client for the main process
    main_client  = SuperClient(server_address)
    # Example: Call additional RPC with the super client
    
    response = main_client.get_batch_status(123, 'example_dataset')
    print(f"Process Main: GetBatchStatus Response - {response}")
    
    # Fork child processes
    for i in range(0, num_processes):
        process = multiprocessing.Process(target=worker, args=(i, server_address))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
   
if __name__ == '__main__':
    test_connections()