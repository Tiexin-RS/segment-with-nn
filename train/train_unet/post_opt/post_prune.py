
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
import os
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
from segelectri.train.train_routine import TrainRoutine
from segelectri.model.unet.unet_t import Unet, get_cluster_unet, get_ordinary_unet, get_prunable_unet
from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds

def save_model_file(model):
  _, keras_file = tempfile.mkstemp('.h5') 
  model.save(keras_file, include_optimizer=False)
  return keras_file

def get_gzipped_model_size(model,prefix = None):
  # It returns the size of the gzipped model in bytes.
  import os
  import zipfile

  keras_file = save_model_file(model)

  _, zipped_file = tempfile.mkstemp(prefix= prefix,suffix='.zip',dir='./prunable_model')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
  return os.path.getsize(zipped_file)


model_path = '/opt/segelectri/train/train_unet/normal_function_model/saved_model/1'
# base_model = tf.keras.models.load_model(filepath=model_path,
#                                         custom_objects={
#                                             'MeanIou': MeanIou,
#                                             'FocalLoss': FocalLoss,
#                                             'LovaszLoss': LovaszLoss,
#                                             'DiceLoss': DiceLoss,
#                                             'BoundaryLoss': BoundaryLoss
#                                         })
base_model = get_ordinary_unet(depth = 3)
base_model.summary()

print("Size of gzipped base_model: %.2f bytes" 
      % (get_gzipped_model_size(base_model,prefix = 'base_model')))
  

def apply_pruning_to_conv2D(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule=pruning_sched.ConstantSparsity(0.8,0))
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_conv2D,
)
# USE ABOVE CODE BLOCK IFF WANT TO DO PRE_LOAD WEIGHT TRANING
# however,it seems not so wise to implement it?

# model_for_pruning = get_prunable_unet(depth = 3)

model_for_pruning.summary()

with tf.device('/cpu:0'):
        ds = get_tr_ds(
            original_pattern='/opt/dataset/tr2_cropped/data/*.png',
            mask_pattern='/opt/dataset/tr2_cropped/label/*.png',
            batch_size=1)

        def reshape_fn(d, l):
            d = tf.cast(tf.reshape(d, (-1, 1024, 1024, 3)), tf.float32) / 255.0
            l = l[:,:,:,1]
            l = tf.one_hot(l, 4)
            l = tf.reshape(l, (-1, 1024, 1024, 4))
            return d, l

        ds = ds.map(reshape_fn)

log_dir = tempfile.mkdtemp(dir='./prunable_model')
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
]

model_for_pruning.compile(optimizer = tf.keras.optimizers.Adam(),
        loss = LovaszLoss(),
        metrics = [MeanIou(num_classes=4)])
model_for_pruning.fit(ds,epochs=1,callbacks = callbacks)

final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
final_model.summary()

# tf.keras.Model.save(final_model,'/opt/segelectri/train/train_unet/normal_function_model/pruned_pb_model/1')

print("final model")
final_model.summary()
print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(model_for_pruning,prefix = 'model_for_pruning')))
print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size(final_model,prefix = 'final_pruned_model')))