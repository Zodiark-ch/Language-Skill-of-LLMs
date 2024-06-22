import pickle, json, decimal, math,os
import numpy as np
from typing import Union
from dataset.ioi_dataset import IOIDataset


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



def get_datasets():
    """from unity"""
    batch_size = 500
    orig = "When John and Mary went to the store, John gave a bottle of milk to Mary"
    new = "When Alice and Bob went to the store, Charlie gave a bottle of milk to Mary"
    prompts_orig = [
        {"S": "John", "IO": "Mary", "TEMPLATE_IDX": -42, "text": orig}
    ]  # TODO make ET dataset construction not need TEMPLATE_IDX
    prompts_new = [{"S": "Alice", "IO": "Bob", "TEMPLATE_IDX": -42, "text": new}]
    prompts_new[0]["text"] = new
    dataset_orig = IOIDataset(
        N=batch_size, prompt_type="mixed"
    )  # TODO make ET dataset construction not need prompt_type
    dataset_new = IOIDataset(
        N=batch_size,
        prompt_type="mixed",
        manual_word_idx=dataset_orig.word_idx,
    )
    return dataset_new, dataset_orig
