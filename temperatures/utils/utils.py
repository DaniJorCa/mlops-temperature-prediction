import os
import pickle
import yaml
from box import ConfigBox



def save_pickle_object(obj, obj_name):
    try:
        with open(obj_name, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise Exception(f"Unable to save pickle object {e}")
    

def load_pickle_object(obj_path):
    try:
        with open(obj_path, 'rb') as f:
            obj = pickle.load(f)

        return obj
    except Exception as e:
        raise Exception(f"Unable to load pickle object {e}")
    

def read_yaml_file(file_path: str) -> ConfigBox:
    """
    Read a yaml file and return it.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return ConfigBox(yaml.safe_load(yaml_file))
    except Exception as e:
        raise Exception(f"Not possible read yaml file {file_path}")
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise Exception(f"Not possible write yaml file {file_path}")
