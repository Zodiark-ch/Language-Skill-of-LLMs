import os
import pickle
from typing import List
from dataclasses import field, dataclass
from utils import set_default_to_empty_string

FOLDER_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/XAIofLLMs/results/'


@dataclass
class DeepArgs:
    task_name: str = "attention_analysis"#['general_discovery','attention_analysis'，'circuit_analysis','residual_analysis','bias_analysis','ioi_check']
    case_type: str='srodataset'#'case', or 'ioidataset','srodataset'
    model_name: str = "gpt2xl"#"gptj""gpt2lmheadmodel","gpt1","gptneox"
    device: str = 'cuda:0'
    logs: str='true'
    save_folder: str = os.path.join(FOLDER_ROOT, task_name,model_name)




    def __post_init__(self):
        
        assert self.task_name in ['general_discovery','attention_analysis','circuit_analysis','residual_analysis','bias_analysis','ioi_check']
        assert self.model_name in ["gpt2xl"]
        assert 'cuda:' in self.device
        self.gpu = int(self.device.split(':')[-1])


    def load_result(self):
        with open(self.save_file_name, 'rb') as f:
            return pickle.load(f)

