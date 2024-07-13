import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model
import logging
import json
from tqdm import tqdm
import copy




hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]
torch.cuda.empty_cache()
set_gpu(args.gpu)


def get_logger(filename, verbosity=1, name=None):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, "w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)

        return logger

if args.task_name=='satisfiability_explain':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model=trunk_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        layer=12
        circuit_layer=29
        circuit_num=12*29
        circuit_name=['circuit0','circuit1','circuit2','circuit3','circuit4','circuit5','circuit6','circuit7','circuit8','circuit9','circuit10',\
            'circuit11','circuit12','circuit13','circuit14','circuit15','circuit16','circuit17','circuit18','circuit19','circuit20','circuit21',\
                'circuit22','circuit23','circuit24','circuit25','circuit26','circuit27','circuit28']
        if args.case_type=='srodataset':
            with open('dataset/srodataset.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='ioidataset':
            _,dataset_orig=get_datasets()
            input_text=dataset_orig.sentences
        
            
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
            if args.case_type=='ioidataset':
                input_case=input_text[i]
            print('To record {}-th case'.format(i))
            inputs = tokenizer(input_case, return_tensors="pt")
    
            with torch.no_grad():
                
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                token_length=inputs['input_ids'].size()[-1]
                with open ('json_logs/satisfiability/gpt2xl/'+input_case+'.json','r') as file:
                    case=json.load(file)
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_case+'_logging.log')  
                logger.info('############ CASE TEXT is'+input_case)
                logger.info('############ CASE Prediction is {}'.format(tokenizer.decode(label_ids)))
                logger.info('############ Refined Forward Graph')
                logger.info('****** Layer 1')
                for cn in range(circuit_layer,circuit_num):
                    circuit_one_satisfiability=torch.IntTensor(case['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]).split(29,dim=-1)
                    
                logger.info()