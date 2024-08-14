import os
import json
import torch
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model,assert_model,attention_flag_model
import logging
from tqdm import tqdm
import copy
from demo_representation_vocb import assert_circuits_equal_output
import matplotlib.pyplot as plt
import numpy

# specify the directory you want to read files from
directory = 'dataset/token_by_token_mixture'
circuit_layer=29
circuit_num=12*29    

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
    
    
if args.task_name=='language_skill':
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    check_model=assert_model(args)
    orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    attention_flag=attention_flag_model(args)
    
        # iterate over every file in the directory
    nagetive_metrix=torch.zeros((circuit_num,circuit_num))
    nagetive_num=0
    positive_metrix=torch.zeros((circuit_num,circuit_num))
    positive_num=0
    for filename in tqdm(os.listdir(directory)):
        # check if the file is a JSON file
        if filename.endswith('.json'):
            fullpath = os.path.join(directory, filename)
            # open the JSON file
            with open(fullpath) as f:
                data = json.load(f)
            input_text=filename.split('.json')[0]
            token_length=len(data)
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids_ori=copy.deepcopy(inputs['input_ids'])
            attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
            for t in range(token_length):
                if args.case_type=="previous_token" or args.case_type=="previous_token_2t":
                    r_token=t
                    s_token=t-1
                token_circuit_record=data[t]
                refined_matrix=torch.zeros((circuit_num,circuit_num))
                assert_refined_matrix=torch.zeros((circuit_num,circuit_num))
                for cn in range(circuit_layer,circuit_num):
                    circuit_one_listtype=token_circuit_record[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                    assert_refined_matrix[cn]=torch.IntTensor(circuit_one_listtype)
                    circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                    for layer in range(cn//circuit_layer):
                        reverse_circuit=1-circuit_one_satisfiability[layer]
                        circuit_one_listtype[layer*29:(layer+1)*29]=reverse_circuit
                    
                    refined_matrix[cn]=torch.IntTensor(circuit_one_listtype)
                    
                inputs['input_ids']=input_ids_ori[:,:t+1]
                inputs['attention_mask']=attention_mask_ori[:,:t+1]
                with torch.no_grad():
                    
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    check_model(inputs,label_ids,assert_refined_matrix)
                    if args.case_type=="previous_token_2t":
                        if t>=2:
                            break
                    if t>=1:
                        
                        attention_flag_metric=attention_flag(inputs,assert_refined_matrix,s_token,r_token)
                        attention_flag_metric=attention_flag_metric.cpu()
                    else:
                        attention_flag_metric=torch.ones(1,29)
                        attention_flag_metric[0][0]=0
                        attention_flag_metric[0][13]=0
                        attention_flag_metric[0][26]=0
                        attention_flag_metric[0][27]=0
                        attention_flag_metric[0][28]=0
                        attention_flag_metric=attention_flag_metric.repeat(12,1)
                    attention_flag_metric=attention_flag_metric.view(-1).unsqueeze(-1)
                    attention_flag_metric=attention_flag_metric.repeat(1,circuit_num)
                    refined_matrix=torch.where(attention_flag_metric>0,refined_matrix,attention_flag_metric)
                    if args.case_type=="previous_token" or args.case_type=="previous_token_2t":
                        if t==0:
                            nagetive_metrix=nagetive_metrix+refined_matrix
                            nagetive_num+=1
                        else: 
                            positive_metrix=positive_metrix+refined_matrix
                            positive_num+=1
    nagetive_metrix=nagetive_metrix/nagetive_num
    positive_metrix=positive_metrix/positive_num
    skill_metrix=positive_metrix-nagetive_metrix
    skill_metrix=torch.where(skill_metrix>0,skill_metrix,0)
                        
    plt.imshow(positive_metrix, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='Color scale')  


    plt.xlabel('N dimension')
    plt.ylabel('M dimension')
    plt.savefig('paper_figure/positive.jpg')
    
    plt.imshow(nagetive_metrix, cmap='gray', vmin=0, vmax=1)
 


    plt.xlabel('N dimension')
    plt.ylabel('M dimension')
    plt.savefig('paper_figure/nagetive.jpg')
    
    plt.imshow(skill_metrix, cmap='gray', vmin=0, vmax=1)
 


    plt.xlabel('N dimension')
    plt.ylabel('M dimension')
    plt.savefig('paper_figure/skill_metrix.jpg')

    plt.show()
    skill_metrix=skill_metrix.numpy().tolist()
    with open('json_logs/language_skill/'+input_text+'.json','w',encoding='utf-8') as dd:
                json.dump(skill_metrix,dd,ensure_ascii=False,sort_keys=True)
                    