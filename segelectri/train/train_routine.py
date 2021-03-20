# coding=utf-8
import os
import json
import logging
import tensorflow as tf

from tensorflow import keras


class TrainRoutine:

    def __init__(self,
                 ds: tf.data.Dataset,
                 model: keras.Model,
                 eval_ds: tf.data.Dataset = None) -> None:
        self.dataset = ds
        self.model = model
        self.eval_dataset = eval_ds

    @staticmethod
    def _ensure_dir_exists(dir):
        ''' ensure dir exists, if not exists create one, otherwise clean it
        '''
        try:
            if os.path.exists(dir):
                os.removedirs(dir)
            os.makedirs(dir, exist_ok=True)
        except OSError as error:
            logging.error(f'failed to create directory {dir}')

    @staticmethod
    def _make_needed_dirs(exp_dir):
        tensorboard_dir = os.path.join(exp_dir, 'events')
        saved_model_dir = os.path.join(exp_dir, 'saved_model')
        config_dir = os.path.join(exp_dir, 'config')
        log_dir = os.path.join(exp_dir, 'log')

        for d in [tensorboard_dir, saved_model_dir, config_dir, log_dir]:
            TrainRoutine._ensure_dir_exists(d)

        return tensorboard_dir, saved_model_dir, config_dir, log_dir

    def run(self, exp_dir, epochs, batch_size=None):
        tensorboard_dir, saved_model_dir, config_dir, log_dir = TrainRoutine._make_needed_dirs(
            exp_dir=exp_dir)
        fh = logging.FileHandler(os.path.join(log_dir,
                                             
                                              'syslog')).setLevel(logging.DEBUG)
        tf.get_logger().addHandler(fh)

        cbs = [keras.callbacks.TensorBoard(tensorboard_dir)]
        if batch_size:
            ds = self.dataset.batch(batch_size=batch_size)
        else:
            ds = self.dataset
        self.model.fit(ds, epochs=epochs, callbacks=cbs)

        self.model.save(saved_model_dir)

        model_config = self.model.get_config()
        train_config = {'epochs': epochs, 'batch_size': batch_size}
        with open(os.path.join(config_dir, 'exp_config.json'), 'w') as outfile:
            exp_config = {
                'model_config': model_config,
                'train_config': train_config,
            }
            json.dump(exp_config, outfile)
