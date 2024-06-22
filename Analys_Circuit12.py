import torch
from args import DeepArgs
from utils import set_gpu,get_datasets
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_into_ebeddingspace import attention_circuit,ioi_attention_circuit
import logging

hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

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

if args.task_name=='attention_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='case':
            model=attention_circuit(args)
            input_text="The Space Needle is in downtown" 
            inputs = tokenizer(input_text, return_tensors="pt")
            model(inputs)
        
        if args.case_type=='ioidataset':
            #ioidataset provides visulization of function heads
            model=ioi_attention_circuit(args)
            _,dataset_orig=get_datasets()
            input_text=dataset_orig.sentences
            word_idx=dataset_orig.word_idx
            duplicate_weight_all=torch.zeros((12,12))
            induction_weight_all=torch.zeros((12,12))
            previous_weight_all=torch.zeros((12,12))
            Name_weight_all=torch.zeros((12,12))
            for i in range (len(input_text)):
                input_case=input_text[i]
                inputs = tokenizer(input_case, return_tensors="pt")
                IO=word_idx['IO'][i]
                IOm1=word_idx['IO-1'][i]
                IOa1=word_idx['IO+1'][i]
                S=word_idx['S'][i]
                Sm1=word_idx['S-1'][i]
                Sa1=word_idx['S+1'][i]
                S2=word_idx['S2'][i]
                with torch.no_grad():
                    duplicate_weight,induction_weight,previous_weight,Name_weight=model(inputs,input_text[i],word_idx,IO,IOm1,IOa1,S,Sm1,Sa1,S2)
                duplicate_weight_all=duplicate_weight_all+duplicate_weight
                induction_weight_all=induction_weight_all+induction_weight
                previous_weight_all=previous_weight_all+previous_weight
                Name_weight_all=Name_weight_all+Name_weight
            duplicate_weight_all=duplicate_weight_all/500
            induction_weight_all=induction_weight_all/500
            previous_weight_all=previous_weight_all/500
            Name_weight_all=Name_weight_all/500
            logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
            logger.info('The duplicate_weight matrix is {}'.format(duplicate_weight_all))
            logger.info('The induction_weight matrix is {}'.format(induction_weight_all))
            logger.info('The previous_weight matrix is {}'.format(previous_weight_all))
            logger.info('The name_weight matrix is {}'.format(Name_weight_all))
            
            
