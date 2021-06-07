import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
import numpy as np
#path of the directory where you want to save your model
frozen_out_path = './frozen'
# name of the .pb file
frozen_graph_filename = 'frozen_model_fixed'
model = tf.keras.models.load_model(filepath='/opt/segelectri/train/train_unet/normal_function_model/clustered_model/1',
                                    custom_objects={
                                        'MeanIou': MeanIou,
                                        'FocalLoss': FocalLoss,
                                        'LovaszLoss': LovaszLoss,
                                        'DiceLoss': DiceLoss,
                                        'BoundaryLoss': BoundaryLoss
                                    })
# Your model
# Convert Keras model to ConcreteFunction
shape = [1] + model.inputs[0].shape[1:]
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(shape, model.inputs[0].dtype))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)

frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)