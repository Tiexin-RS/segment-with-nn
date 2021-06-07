from grpc.beta import implementations
import tensorflow as tf
import numpy as np
import time
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import http.client
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(loglevel):
    # the file handler receives all messages from level DEBUG on up, regardless
    fileHandler = TimedRotatingFileHandler(
        filename="./logging_grpc.log",
        when="midnight"
    )
    fileHandler.setLevel(logging.DEBUG)
    handlers = [fileHandler]

    if loglevel is not None:
        # if a log level is configured, use that for logging to the console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(loglevel)
        handlers.append(stream_handler)

    if loglevel == logging.DEBUG:
        # when logging at debug level, make http.client extra chatty too
        # http.client *uses `print()` calls*, not logging.
        http.client.HTTPConnection.debuglevel = 1

    # finally, configure the root logger with our choice of handlers
    # the logging level of the root set to DEBUG (defaults to WARNING otherwise).
    logformat = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(
        format=logformat, datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers, level=logging.DEBUG
    )

http_client_logger = logging.getLogger("http.client")

def print_to_log(*args):
    http_client_logger.debug(" ".join(args))

# monkey-patch a `print` global into the http.client module; all calls to
# print() in that module will then use our print_to_log implementation
http.client.print = print_to_log

level = logging.DEBUG
setup_logging(level)

channel = implementations.insecure_channel('localhost', 8500)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'deeplab_53_unfreeze'
request.model_spec.signature_name = 'serving_default'

x_data = np.load('/home/Tiexin-RS/code/workspace/wjz/segment-with-nn/serving/tf_serving/a.npy')
request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(x_data, dtype=tf.float32))

# start = time.time()
result = stub.Predict(request, 10.0)
# result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout
# result = result_future.result()
# stop = time.time()
# print('time is ', stop - start)

# outputs_tensor_proto = result.outputs["output_1"]
# shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
# outputs = tf.constant(outputs_tensor_proto.float_val, shape=shape)
# tf.print(outputs.shape)