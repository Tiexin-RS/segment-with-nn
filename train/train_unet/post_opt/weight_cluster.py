
from segelectri.model.unet.unet_t import get_ordinary_unet
from sys import prefix
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
import os
from segelectri.loss_metrics.loss import FocalLoss, LovaszLoss, DiceLoss, BoundaryLoss
from segelectri.loss_metrics.metrics import MeanIou
from segelectri.train.train_routine import TrainRoutine
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

  _, zipped_file = tempfile.mkstemp(prefix = prefix,suffix='.zip',dir='./clustered_model')
  print("save "+ prefix + "model")
  # tf.keras.models.save_model(model,zipped_file,include_optimizer=False)
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
  return os.path.getsize(zipped_file)


model_path = '/opt/segelectri/train/train_unet/normal_function_model/saved_model/1'
base_model = tf.keras.models.load_model(filepath=model_path,
                                        custom_objects={
                                            'MeanIou': MeanIou,
                                            'FocalLoss': FocalLoss,
                                            'LovaszLoss': LovaszLoss,
                                            'DiceLoss': DiceLoss,
                                            'BoundaryLoss': BoundaryLoss
                                        })
# base_model = get_ordinary_unet(depth = 3)
print("Size of gzipped base_model: %.2f bytes" 
      % (get_gzipped_model_size(base_model,prefix = 'base_model')))

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
  'number_of_clusters': 256,
  'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
}

def apply_clustering_to_conv2D(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return cluster_weights(layer, **clustering_params)
  return layer

clustered_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_clustering_to_conv2D,
)

clustered_model.summary()

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

clustered_model.compile(optimizer = tf.keras.optimizers.Adam(),
        loss = LovaszLoss(),
        metrics = [MeanIou(num_classes=4)])
clustered_model.fit(ds,epochs=1)

final_model = tfmot.clustering.keras.strip_clustering(clustered_model)

print("final model")
final_model.summary()
print("Size of gzipped clustered model without stripping: %.2f bytes" 
      % (get_gzipped_model_size(clustered_model,prefix = 'model_for_clustering')))
print("Size of gzipped clustered model with stripping: %.2f bytes" 
      % (get_gzipped_model_size(final_model,prefix = 'final_clustered_model')))
