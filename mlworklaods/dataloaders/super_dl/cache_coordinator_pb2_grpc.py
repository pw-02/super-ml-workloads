# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import mlworklaods.dataloaders.super_dl.cache_coordinator_pb2 as proto_dot_cache__coordinator__pb2


class CacheCoordinatorServiceStub(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./proto/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/CacheCoordinatorService/Ping',
                request_serializer=proto_dot_cache__coordinator__pb2.PingRequest.SerializeToString,
                response_deserializer=proto_dot_cache__coordinator__pb2.PingResponse.FromString,
                )
        self.RegisterJob = channel.unary_unary(
                '/CacheCoordinatorService/RegisterJob',
                request_serializer=proto_dot_cache__coordinator__pb2.RegisterJobRequest.SerializeToString,
                response_deserializer=proto_dot_cache__coordinator__pb2.RegisterJobResponse.FromString,
                )
        self.GetNextBatchToProcess = channel.unary_unary(
                '/CacheCoordinatorService/GetNextBatchToProcess',
                request_serializer=proto_dot_cache__coordinator__pb2.GetNextBatchRequest.SerializeToString,
                response_deserializer=proto_dot_cache__coordinator__pb2.GetNextBatchResponse.FromString,
                )
        self.JobEnded = channel.unary_unary(
                '/CacheCoordinatorService/JobEnded',
                request_serializer=proto_dot_cache__coordinator__pb2.JobEndedRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GetDatasetInfo = channel.unary_unary(
                '/CacheCoordinatorService/GetDatasetInfo',
                request_serializer=proto_dot_cache__coordinator__pb2.DatasetInfoRequest.SerializeToString,
                response_deserializer=proto_dot_cache__coordinator__pb2.DatasetInfoResponse.FromString,
                )


class CacheCoordinatorServiceServicer(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./proto/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    def Ping(self, request, context):
        """rpc RegisterJob(JobInfo) returns (RegisterJobResponse);
        rpc SendMetrics(MetricsRequest) returns (google.protobuf.Empty);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterJob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetNextBatchToProcess(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def JobEnded(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDatasetInfo(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CacheCoordinatorServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=proto_dot_cache__coordinator__pb2.PingRequest.FromString,
                    response_serializer=proto_dot_cache__coordinator__pb2.PingResponse.SerializeToString,
            ),
            'RegisterJob': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterJob,
                    request_deserializer=proto_dot_cache__coordinator__pb2.RegisterJobRequest.FromString,
                    response_serializer=proto_dot_cache__coordinator__pb2.RegisterJobResponse.SerializeToString,
            ),
            'GetNextBatchToProcess': grpc.unary_unary_rpc_method_handler(
                    servicer.GetNextBatchToProcess,
                    request_deserializer=proto_dot_cache__coordinator__pb2.GetNextBatchRequest.FromString,
                    response_serializer=proto_dot_cache__coordinator__pb2.GetNextBatchResponse.SerializeToString,
            ),
            'JobEnded': grpc.unary_unary_rpc_method_handler(
                    servicer.JobEnded,
                    request_deserializer=proto_dot_cache__coordinator__pb2.JobEndedRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GetDatasetInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDatasetInfo,
                    request_deserializer=proto_dot_cache__coordinator__pb2.DatasetInfoRequest.FromString,
                    response_serializer=proto_dot_cache__coordinator__pb2.DatasetInfoResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'CacheCoordinatorService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CacheCoordinatorService(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./proto/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/Ping',
            proto_dot_cache__coordinator__pb2.PingRequest.SerializeToString,
            proto_dot_cache__coordinator__pb2.PingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/RegisterJob',
            proto_dot_cache__coordinator__pb2.RegisterJobRequest.SerializeToString,
            proto_dot_cache__coordinator__pb2.RegisterJobResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetNextBatchToProcess(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/GetNextBatchToProcess',
            proto_dot_cache__coordinator__pb2.GetNextBatchRequest.SerializeToString,
            proto_dot_cache__coordinator__pb2.GetNextBatchResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def JobEnded(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/JobEnded',
            proto_dot_cache__coordinator__pb2.JobEndedRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDatasetInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/GetDatasetInfo',
            proto_dot_cache__coordinator__pb2.DatasetInfoRequest.SerializeToString,
            proto_dot_cache__coordinator__pb2.DatasetInfoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
