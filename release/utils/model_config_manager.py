import contextlib
import os
import json
from typing import List

import tensorflow as tf
from tensorflow_serving.config.model_server_config_pb2 import ModelConfigList, ModelConfig


class ModelConfigManager(object):

    def __init__(self, model_config_dir: str) -> None:
        if not tf.io.gfile.exists(model_config_dir):
            tf.io.gfile.makedirs(model_config_dir)
        self.model_config_file = os.path.join(model_config_dir, 'models.config')
        self.registy_file = os.path.join(model_config_dir, 'registry.json')

    def _load_latest_model_config(self) -> ModelConfigList:
        if tf.io.gfile.exists(self.model_config_file):
            with tf.io.gfile.GFile(self.model_config_file, mode='rb') as f:
                return ModelConfigList.FromString(f.read())
        else:
            return ModelConfigList()

    def _sync_model_config_to_fs(self, model_configs: ModelConfigList):
        with tf.io.gfile.GFile(self.model_config_file, mode='wb') as f:
            f.write(model_configs.SerializeToString())

    def _load_latest_registry(self) -> dict:
        if tf.io.gfile.exists(self.registy_file):
            with tf.io.gfile.GFile(self.registy_file, 'r') as f:
                return json.load(f)
        else:
            return {}

    def _sync_registry_to_fs(self, registry: dict):
        with tf.io.gfile.GFile(self.registy_file, mode='w') as f:
            json.dump(registry, f)

    @staticmethod
    def _get_model_versions(model_config: ModelConfig) -> List[int]:
        return model_config.model_version_policy.specific.versions

    @contextlib.contextmanager
    def model_config_session_scope(self):
        try:
            model_configs, registry = self._load_latest_model_config(), self._load_latest_registry()
            yield model_configs, registry
        finally:
            self._sync_model_config_to_fs(model_configs), self._sync_registry_to_fs(registry)

    def register(self, model_name: str):
        with self.model_config_session_scope() as (model_configs, registry):
            if model_name in registry:
                raise ValueError(f'Already registered model {model_name}')
            registry[model_name] = {
                'versions': []
            }

    def update(self, model_name: str, model_base_path: str, model_version: str):
        with self.model_config_session_scope() as (model_configs, registry):
            if model_name not in registry:
                raise ValueError(f'Unregistered Model {model_name}')
            for config in model_configs.config:
                model_versions = self._get_model_versions(config)
                if model_name == config.name:
                    assert model_base_path == config.base_path
                    if model_version in model_versions:
                        raise ValueError(
                            f'Duplicated VersionedModel {model_name}[{model_version}]'
                        )
                    model_versions.append(model_version)
                    registry[model_name]['versions'].append(model_version)
                    return
            config = model_configs.config.add()
            config.name = model_name
            config.model_version_policy.specific.versions.extend([model_version])
            config.base_path = model_base_path
            config.model_platform = 'tensorflow'

    def delete(self, model_name: str, model_version: int):
        with self.model_config_session_scope() as (model_configs, registry):
            if model_name not in registry:
                raise ValueError(f'Unregistered Model {model_name}')
            for i, config in enumerate(model_configs.config):
                model_versions = self._get_model_versions(config)
                if config.name == model_name:
                    if model_version == -1:
                        model_configs.config.pop(i)
                        registry.pop(model_name)
                        return
                    if model_version not in model_versions:
                        break
                    if len(model_versions) == 1:
                        model_configs.config.pop(i)
                        registry[model_name]['versions'] = []
                    else:
                        model_versions.remove(model_version)
                        registry[model_name]['versions'].remove(model_version)
                    return
            if model_name in registry:
                registry.pop(model_name)

    def list(self) -> dict:
        with self.model_config_session_scope() as (_, registry):
            return registry
