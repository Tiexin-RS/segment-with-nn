import os
import logging

import tensorflow as tf

class FileManager(object):
    def __init__(self, model_storage_dir: str) -> None:
        if not tf.io.gfile.exists(model_storage_dir):
            logging.info(f'make model storage directory {model_storage_dir}')
            tf.io.gfile.makedirs(model_storage_dir)
        self.model_storage_dir = model_storage_dir

    def copy_model_to_serving(self, model_base_path: str,
                              model_name: str, model_version: int) -> str:
        logging.debug(f'model_path is {model_base_path}')
        serving_side_model_path = os.path.join(self.model_storage_dir, model_name, str(model_version))
        if not tf.io.gfile.exists(serving_side_model_path):
            tf.io.gfile.makedirs(serving_side_model_path)
        for root, dirs, files in tf.io.gfile.walk(model_base_path):
            relative = os.path.relpath(root, model_base_path)
            for f in files:
                tf.io.gfile.copy(
                    os.path.join(root, f),
                    os.path.join(serving_side_model_path, relative, f), overwrite=True)
            for d in dirs:
                tf.io.gfile.makedirs(os.path.join(serving_side_model_path, relative, d))

        return os.path.join(self.model_storage_dir, model_name)

    def delete_model_from_serving(self, model_name: str, model_version: int):
        serving_side_model_path = os.path.join(self.model_storage_dir, model_name, str(model_version))
        if tf.io.gfile.exists(serving_side_model_path):
            tf.io.gfile.rmtree(serving_side_model_path)