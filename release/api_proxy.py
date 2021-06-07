import json
import os
import tempfile
import logging
import traceback

from utils.model_config_manager import ModelConfigManager
from utils.file_manager import FileManager 

def api_proxy(action: str, kwargs: dict) -> str:
    serving_storage_dir = os.getenv('SERVING_STORAGE_DIR', tempfile.gettempdir())
    logging.info(f'serving_storage_dir is {serving_storage_dir}')
    model_config_dir = os.path.join(serving_storage_dir, 'configs')
    model_storage_dir = os.path.join(serving_storage_dir, 'models')
    model_config_manager = ModelConfigManager(model_config_dir)
    file_manager = FileManager(model_storage_dir)
    
    try:
        if action == 'list':
            configs = model_config_manager.list()
            return json.dumps(configs)
        elif action == 'register':
            model_config_manager.register(**kwargs)
        elif action == 'update':
            model_base_path = file_manager.copy_model_to_serving(**kwargs)
            kwargs['model_base_path'] = model_base_path
            model_config_manager.update(**kwargs)
        elif action == 'delete':
            file_manager.delete_model_from_serving(**kwargs)
            model_config_manager.delete(**kwargs)
        return 'Done.'
    except Exception as err:
        return f'Failed to {action}, because {err}, {traceback.format_exc()}'
