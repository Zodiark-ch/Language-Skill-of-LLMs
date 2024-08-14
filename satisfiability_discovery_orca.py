import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model,assert_model
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

if args.task_name=='satisfiability_discovery':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model=trunk_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        #assert_model=refine_explain_model(args)
        layer=12
        circuit_layer=29
        circuit_num=12*29
        if args.case_type=='srodataset':
            with open('dataset/srodataset.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='ioidataset':
            with open('dataset/ioidataset.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orcadataset':
            with open('dataset/OpenOrcadataset.json','r') as f: 
                input_text=json.load(f)
        
            
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
            if args.case_type=='ioidataset':
                input_case=input_text[i]['text']
            if args.case_type=='orcadataset':
                input_case=input_text[i]['text']
            print('To record {}-th case'.format(i))
            inputs = tokenizer(input_case, return_tensors="pt")
    
            with torch.no_grad():
                
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                token_length=inputs['input_ids'].size()[-1]
                branch_cut=torch.zeros((circuit_num,circuit_num))
                
                #init the input matrix 
                token_num=inputs['input_ids'].size()[-1]
                input_matrix=torch.zeros((12,29,token_num,768)).cuda()
                cut_circuit_tensor_all=None
                top_token,input_matrix,cut_circuit_tensor_all=model(inputs,label_ids,0,0,input_matrix,cut_circuit_tensor_all)
                assert top_token[0].item()==label_ids.item()
                for m in tqdm(range(circuit_num)):
                    for n in range(circuit_num):
                        if m//circuit_layer > n//circuit_layer and (m+1)%29!=27 and (m+1)%29!=28 and (m+1)%29!=0 and branch_cut[m][n]!=1:
                            branch_cut[m][n]=1
                            
                            top_token,input_matrix_new,cut_circuit_tensor_all=model(inputs,label_ids,m,n,input_matrix,cut_circuit_tensor_all)
                            if top_token[0].item()!=label_ids.item():
                                branch_cut[m][n]=0
                            else:
                                #assert_model(inputs,label_ids,branch_cut)
                                input_matrix=input_matrix_new
                            torch.cuda.empty_cache()
                            
                                
            
            logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'/'+input_case+'_logging.log')  
            all_branch_cut=[]
            for id in range(29,348):
                    all_branch_cut_dict={}
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
                    all_branch_cut_dict['layer {} and circuit {}'.format(id//circuit_layer,id%circuit_layer)]=branch_cut[id].tolist()
                    all_branch_cut.append(all_branch_cut_dict)
            with open('json_logs/satisfiability/gpt2xl/'+args.case_type+'/'+input_case+'.json','w',encoding='utf-8') as data:
                json.dump(all_branch_cut,data,ensure_ascii=False,sort_keys=True)
            logging.shutdown()                       
                    