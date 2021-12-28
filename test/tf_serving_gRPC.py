import grpc
import time
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def print_performance(host:str, port:int, model:str, max_iter:int=10)->None:
    
    # test image set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = np.expand_dims(test_images/255.0, -1).astype(np.float32)
    

    
    response_times = list()
    for idx in range(max_iter):
        start_time = time.time()
        channel = grpc.insecure_channel(f'{host}:{port}')
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            
        # Send request
        # See prediction_service.proto for gRPC request/response details.
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_1'].CopyFrom(
            tf.make_tensor_proto(test_images))
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        response_times.append(time.time() - start_time)
        result = result.outputs['output_1'].float_val
        #print(len(result), result[0])    
        
    print(f'AVG = {np.average(response_times):>.3f} s, STD = ±{np.std(response_times):>.3f} s')
    
if __name__ == '__main__':
    print_performance(host='localhost', port=8500, model='mnist_model')