import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import assert_model,refined_explain_model,ioi_check_model,common_graph_model
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
    
    


if args.task_name=='distillation':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        check_model=assert_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        explain_model=refined_explain_model(args)
        distillation_model=common_graph_model(args)

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
            with open('dataset/ioidataset.json','r') as f: 
                input_text=json.load(f)
        
        ori_accuracy=0  
        new_accuracy=0  
        consistency=0
        refined_matrix_common=torch.ones((12,circuit_layer))
        refined_matrix_common=refined_matrix_common.to(int)
        #get common graph
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
                label=input_text[i]['attribute']
            if args.case_type=='ioidataset':
                input_case=input_text[i]['text']
                IO=input_text[i]['IO']
                IO_m1=input_text[i]['IO-1']
                IO_a1=input_text[i]['IO+1']
                S=input_text[i]['S']
                S_m1=input_text[i]['S-1']
                S_a1=input_text[i]['S+1']
                S2=input_text[i]['S2']
                end=input_text[i]['end']
            print('To record {}-th case'.format(i))
            inputs = tokenizer(input_case, return_tensors="pt")
            with torch.no_grad():
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                ori_token=tokenizer.decode(label_ids).split(' ')[-1]
            if i==5:
                break
            #if ori_token==label: 
            with open ('json_logs/satisfiability/gpt2xl/'+args.case_type+'/'+input_case+'.json','r') as file:
                case=json.load(file)
            refined_matrix=torch.zeros((12,29))
            
            for cn in range(circuit_layer,circuit_num):
                circuit_one_listtype=case[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype)
                if torch.sum(circuit_one_satisfiability).item()==(cn//circuit_layer)*29:
                    refined_matrix[cn//circuit_layer][cn%circuit_layer]=1
                

            refined_matrix=refined_matrix.to(int)
            refined_matrix_common=refined_matrix_common&refined_matrix
            print(torch.sum(refined_matrix_common).item())
        common_graph=torch.zeros((circuit_num,circuit_num))
        for m in range(12):
            for n in range(29):
                if refined_matrix_common[m][n]==1:
                    delete_num=m*29
                    for d in range(delete_num):
                        common_graph[m*29+n][d]=1
        
        
        for i in range (200,400):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
                label=input_text[i]['attribute']
            if args.case_type=='ioidataset':
                input_case=input_text[i]['text']
                IO=input_text[i]['IO']
                IO_m1=input_text[i]['IO-1']
                IO_a1=input_text[i]['IO+1']
                S=input_text[i]['S']
                S_m1=input_text[i]['S-1']
                S_a1=input_text[i]['S+1']
                S2=input_text[i]['S2']
                end=input_text[i]['end']
            print('To record {}-th case'.format(i))
            inputs = tokenizer(input_case, return_tensors="pt")
            with torch.no_grad():
                
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                new_label_ids=distillation_model(inputs,common_graph)
                if label_ids.item()==new_label_ids.item():
                    consistency+=1
                ori_token=tokenizer.decode(label_ids).split(' ')[-1]
                new_token=tokenizer.decode(new_label_ids).split(' ')[-1]
                
                
            print(consistency)
                    
                    
                        
                        
                
                
                