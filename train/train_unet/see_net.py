import tensorflow as tf
from tensorflow.lite.python import convert
from tensorflow.python.keras.metrics import Precision
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
import tensorflow_model_optimization as tfmot
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import pathlib

if __name__ == "__main__":
    model = tf.keras.models.load_model("../train_unet/exp/37/saved_model",custom_objects={'LovaszLoss': LovaszLoss})
    model.summary()

    """
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode = trt.TrtPrecisionMode.FP32)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir='../train_unet/exp/36/saved_model',conversion_params=params)
    converter.convert()
    converter.save('trt_savedmodel')
    before:2922881
    after:127084028
    """

    """
    converter = tf.lite.TFLiteConverter.from_saved_model('../train_unet/exp/36/saved_model')
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("/tmp/unet_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"mnist_model.tflite"
    origin_byte = tflite_model_file.write_bytes(tflite_model)
    print(origin_byte)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"unet_model_quant.tflite"
    after_byte = tflite_model_file.write_bytes(tflite_quant_model)
    print(after_byte)
    before:2922881
    after:127084028
    """

    """
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params._replace(precision_mode = trt.TrtPrecisionMode.INT8)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir='../train_unet/exp/37/saved_model',conversion_params=params)
    converter.convert()
    converter.save('trt_savedmodel')
    before:2809876
    after:127003352
    """

    
    """converter = tf.lite.TFLiteConverter.from_saved_model('../train_unet/exp/37/saved_model')
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./tmp/unet_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"unet_model.tflite"
    origin_byte = tflite_model_file.write_bytes(tflite_model)
    print(origin_byte)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"unet_model_quant.tflite"
    after_byte = tflite_model_file.write_bytes(tflite_quant_model)
    print(after_byte)"""
    