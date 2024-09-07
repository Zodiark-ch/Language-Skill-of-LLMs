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
import numpy as np
from plot import plot_cka_matrix,annotate_heatmap
from sklearn.cluster import KMeans
from matplotlib import pyplot
import shutil
import seaborn as sns
import torch.nn.functional as F
# specify the directory you want to read files from
directory = 'json_logs/token_by_token/gpt2xl/icl_sst2_cluster0'
filter_weight=0.6

    
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
    
    attention_flag_metric_default=torch.ones(circuit_num,circuit_num)
    attention_flage_metric_mask=torch.zeros(circuit_num,circuit_num)
    for i in range(circuit_num):
        for j in range(circuit_num):
            if (i+1)%29==27 or (i+1)%29==28 or (i+1)%29==0 or (i+1)%29==1 or (i+1)%29==14  or (j+1)%29==27 or (j+1)%29==28 or (j+1)%29==0 or (j+1)%29==1:
                attention_flag_metric_default[i][j]=0
    
        # iterate over every file in the directory
    negative_metrix=torch.zeros((circuit_num,circuit_num))
    nagetive_num=0
    positive_metrix=torch.zeros((circuit_num,circuit_num))
    positive_num=0
    self_metrix=torch.zeros((circuit_num,circuit_num))
    self_num=0
 
    for filename in tqdm(os.listdir(directory)):
        # check if the file is a JSON file
        if filename.endswith('.json'):
            fullpath = os.path.join(directory, filename)
            # open the JSON file
            with open(fullpath) as f:
                data = json.load(f)
        
            with open('dataset/icl_sst2.json') as icldataset:
                icl_data=json.load(icldataset)
            if filename=='self.json':
                continue
            fileid=int(filename.split('.')[0])
            assert icl_data[fileid]['id']==fileid
            input_text= icl_data[fileid]['text']
            background_text=icl_data[fileid]['question']
            circuit_length=len(data)
            if circuit_length!=3:
                continue
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids_ori=copy.deepcopy(inputs['input_ids'])
            token_length=input_ids_ori.size()[-1]
            attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
            outputs = orig_model(**inputs, labels=inputs["input_ids"])
            _,label_ids=torch.topk(outputs.logits[0][-1],1)
            
            for t in range(circuit_length):
                
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
                if t==0:  
                    inputs = tokenizer(background_text, return_tensors="pt")              
                if t==1:
                    inputs=inputs
                    
                if t==2:
                    inputs['input_ids']=input_ids_ori[:,-1].unsqueeze(0)
                    inputs['attention_mask']=attention_mask_ori[:,-1]
                with torch.no_grad():
                    
                    
                    attention_flag_metric=attention_flag_metric_default
                    refined_matrix=torch.where(attention_flag_metric>0,refined_matrix,attention_flag_metric)
                    #print(refined_matrix[29][:29])
                    
                    if t==0:
                        negative_metrix=negative_metrix+refined_matrix
                        nagetive_num+=1
                    elif t==1: 
                        positive_metrix=positive_metrix+refined_matrix
                        positive_num+=1
                    else:
                        self_metrix=self_metrix+refined_matrix
                        self_num+=1
                            
    negative_metrix=negative_metrix/nagetive_num
    negative_metrix=torch.where(attention_flag_metric_default>0,negative_metrix,0)
    negative_metrix=negative_metrix.view(-1).unsqueeze(0)
    positive_metrix=positive_metrix/positive_num
    positive_metrix=torch.where(attention_flag_metric_default>0,positive_metrix,0)
    positive_metrix=positive_metrix.view(-1).unsqueeze(0)
    self_metrix=self_metrix/self_num
    self_metrix=torch.where(attention_flag_metric_default>0,self_metrix,0)
    self_metrix=self_metrix.view(-1).unsqueeze(0)
    
    sample_all=torch.cat((negative_metrix,positive_metrix,self_metrix),dim=0)
    sample_all=sample_all.permute(1,0)
    cut_num=0
    for i in range(negative_metrix.size()[1]):
        if sample_all[i-cut_num][0]==sample_all[i-cut_num][1]==sample_all[i-cut_num][2]==0 or sample_all[i-cut_num][0]==sample_all[i-cut_num][1]==0 or sample_all[i-cut_num][2]==sample_all[i-cut_num][1]==0:
            arr1=sample_all[0:i-cut_num]
            arr2=sample_all[i-cut_num+1:]
            sample_all=torch.cat((arr1,arr2),dim=0)
            cut_num+=1
            
    sample05=copy.deepcopy(sample_all)
    cut_num=0
    for i in range(sample_all.size()[0]):
        if sample05[i-cut_num][1]<0.5:
            arr1=sample05[0:i-cut_num]
            arr2=sample05[i-cut_num+1:]
            sample05=torch.cat((arr1,arr2),dim=0)
            cut_num+=1
            
    sample06=copy.deepcopy(sample_all)
    cut_num=0
    for i in range(sample_all.size()[0]):
        if sample06[i-cut_num][1]<0.6:
            arr1=sample06[0:i-cut_num]
            arr2=sample06[i-cut_num+1:]
            sample06=torch.cat((arr1,arr2),dim=0)
            cut_num+=1
            
    sample07=copy.deepcopy(sample_all)
    cut_num=0
    for i in range(sample_all.size()[0]):
        if sample07[i-cut_num][1]<0.7:
            arr1=sample07[0:i-cut_num]
            arr2=sample07[i-cut_num+1:]
            sample07=torch.cat((arr1,arr2),dim=0)
            cut_num+=1
            
    sample08=copy.deepcopy(sample_all)
    cut_num=0
    for i in range(sample_all.size()[0]):
        if sample08[i-cut_num][1]<0.4:
            arr1=sample08[0:i-cut_num]
            arr2=sample08[i-cut_num+1:]
            sample08=torch.cat((arr1,arr2),dim=0)
            cut_num+=1
    #sample_all=F.softmax(sample_all,dim=0)
    sample05=sample05.numpy()
    sample06=sample06.numpy()
    sample07=sample07.numpy()
    sample08=sample08.numpy()
    
    #fig, ax1 = plt.subplots(3,1,sharex=True)
    sns.kdeplot(x=sample08[:,1],y=sample08[:,0],fill=True, cmap='Oranges',levels=[0.35,0.5,0.6,0.7,0.8,0.9,1],shade_lowest=False)
    sns.kdeplot(x=sample05[:,1],y=sample05[:,0],fill=True, cmap='Reds',levels=[0.35,0.5,0.6,0.7,0.8,0.9,1],shade_lowest=False)
    sns.kdeplot(x=sample06[:,1],y=sample06[:,0],fill=True, cmap='Greens',levels=[0.35,0.5,0.6,0.7,0.8,0.9,1],shade_lowest=False)
    sns.kdeplot(x=sample07[:,1],y=sample07[:,0],fill=True, cmap='Blues',levels=[0.35,0.5,0.6,0.7,0.8,0.9,1],shade_lowest=False,cbar=True)
    
    
    
    plt.legend()
    plt.savefig('paper_figure/icl_sst2_token_distribution.jpg')