import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model
import logging
import json
from tqdm import tqdm



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

if args.task_name=='satisfiability_discovery':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model=trunk_model(args)
        layer=12
        circuit_layer=29
        circuit_num=12*29
        if args.case_type=='srodataset':
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            i=0
            
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
        
                with torch.no_grad():
                    orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    token_length=inputs['input_ids'].size()[-1]
                    branch_cut=torch.zeros((circuit_num,circuit_num))
                    top_token=model(inputs,label_ids,branch_cut)
                    assert top_token[0].item()==label_ids.item()
                    for m in tqdm(range(circuit_num)):
                        for n in range(circuit_num):
                            if n//circuit_layer > m//circuit_layer and (n+1)%29!=27 and (n+1)%29!=28 and (n+1)%29!=0:
                                temp_branch=branch_cut
                                temp_branch[n][m]=1
                                top_token=model(inputs,label_ids,temp_branch)
                                if top_token[0].item()==label_ids.item():
                                    
                                    branch_cut=temp_branch
                                
                                    
                
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_text+'_logging.log')  
                for id in range(29,348):
                        branch_cut_id=branch_cut[id].split(29,dim=-1)
                        logger.info('### for layer {} and circuit {}, the cut list of layer 0 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[0]))
                        logger.info('### for layer {} and circuit {}, the cut list of layer 1 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[1])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 2 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[2])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 3 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[3])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 4 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[4])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 5 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[5])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 6 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[6])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 7 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[7])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 8 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[8])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 9 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[9])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 10 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[10])) 
                        logger.info('### for layer {} and circuit {}, the cut list of layer 11 is \n{}'.format(id//circuit_layer,id%circuit_layer,branch_cut_id[11]))   
                logging.shutdown() 
                                
                    