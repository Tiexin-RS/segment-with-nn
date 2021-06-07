# import json
# import tensorflow as tf
import requests
import numpy as np
import time
import http.client
import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(loglevel):
    # the file handler receives all messages from level DEBUG on up, regardless
    fileHandler = TimedRotatingFileHandler(
        filename="./logging.log",
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

url = 'http://localhost:8501/v1/models/deeplab_53_unfreeze:predict'
data = np.load('a.npy')
payload = {"inputs": {'input_1': data.tolist()}}
# start = time.time()
requests.post(url=url, json=payload)
# stop = time.time()
# print('time is ', stop - start)
# print(r.headers)
# pred = json.loads(r.content.decode('utf-8'))
# json_data = json.dumps(pred)
# json_file = open('./pred.json', 'w')
# json_file.write(json_data)
# json_file.close()
