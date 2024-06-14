import pickle, json, decimal, math,os
import numpy as np
from typing import Union



def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js


def set_default_to_empty_string(v, default_v, activate_flag):
    if ((default_v is not None and v == default_v) or (
            default_v is None and v is None)) and (activate_flag):
        return ""
    else:
        return f'_{v}'
    
def set_gpu(gpu_id: Union[str, int]):
    if isinstance(gpu_id, int):
        gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    



def get_model_layer_num(model = None):
    num_layer = None
    if model is not None:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layer = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layers'):
            num_layer = model.config.n_layers
        elif hasattr(model.config, 'n_layer'):
            num_layer = model.config.n_layer
        else:
            pass
    if num_layer is None:
        raise ValueError(f'cannot get num_layer from model: {model} or model_name: {model_name}')
    return num_layer