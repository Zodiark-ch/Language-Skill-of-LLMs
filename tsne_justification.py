
from sklearn.manifold import TSNE
import os
import json
import torch
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model,assert_model,attention_flag_model,cut_prediction
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

# specify the directory you want to read files from
directory = 'json_logs/token_by_token/gpt2xl/all_mix_cluster1_cluster0'
filter_weight=0.4

    
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
    
def get_token_rank(representation,label_ids): 
        ranks = torch.argsort(torch.argsort(representation))
        rank_label_list=ranks[label_ids]
        rank_label_list=representation.size()[-1]-rank_label_list
        return rank_label_list    

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
attention_flag=attention_flag_model(args)
cut_model=cut_prediction(args)

attention_flag_metric_default=torch.ones(circuit_num,circuit_num)
attention_flage_metric_mask=torch.zeros(circuit_num,circuit_num)
for i in range(circuit_num):
    for j in range(circuit_num):
        if i<29 or (i+1)%29==27 or (i+1)%29==28 or (i+1)%29==0 or (i+1)%29==1 or (i+1)%29==14  or (j+1)%29==27 or (j+1)%29==28 or (j+1)%29==0 or (j+1)%29==1:
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
        input_text=filename.split('.json')[0]
        token_length=len(data)
        if token_length<3:
                continue
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids_ori=copy.deepcopy(inputs['input_ids'])
        attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
        
        r_token=1
        s_token=0
        for t in range(token_length):
            token_circuit_record=data[t]
            if t==2:
                token_circuit_record=data[-1]
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
            if t<2:    
                inputs['input_ids']=input_ids_ori[:,:t+1]
                inputs['attention_mask']=attention_mask_ori[:,:t+1]
            if t==2:
                inputs['input_ids']=input_ids_ori[:,-1].unsqueeze(0)
                inputs['attention_mask']=attention_mask_ori[:,-1]
            with torch.no_grad():
                
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                #check_model(inputs,label_ids,assert_refined_matrix)
                
                if t>2:
                    break
                if t==1:
                    
                    attention_flag_metric=attention_flag(inputs,assert_refined_matrix,s_token,r_token)
                    attention_flag_metric=attention_flag_metric.cpu()
                    attention_flag_metric=attention_flag_metric.view(-1).unsqueeze(-1)
                    attention_flag_metric=attention_flag_metric.repeat(1,circuit_num)
                    attention_flag_metric=torch.where(attention_flag_metric_default>0,attention_flag_metric,attention_flag_metric_default)
                else:
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
positive_metrix=positive_metrix/positive_num
self_metrix=self_metrix/self_num
skill_metrix=positive_metrix-self_metrix
skill_metrix=torch.where(skill_metrix>0,skill_metrix,0)
skill_topk=torch.where(skill_metrix>filter_weight,skill_metrix,0)
topk_idx=torch.nonzero(skill_topk)
chain_all=[]
for idx in range(topk_idx.size()[0]):
    r_circuit=topk_idx[idx][0].item()
    s_circuit=topk_idx[idx][1].item()
    print('recieve_circuit: layer {} circuit {},   sent_circuit: layer {} circuit {}  with its weight {}'.format(r_circuit//29, r_circuit%29, s_circuit//29, s_circuit%29,skill_topk[r_circuit][s_circuit]))
    insert_flag=0

    new_chain=[s_circuit,r_circuit]
    chain_all.append(new_chain)
    print('new_chain: ', new_chain)
    for chain in range(len(chain_all)):
        if chain_all[chain][-1]==s_circuit:
            insert_flag=1
            new_chain=copy.deepcopy(chain_all[chain])
            new_chain.append(r_circuit)
            chain_all.append(new_chain)
            print('new_chain: ', new_chain)
        
print('there are all {} chains'.format(len(chain_all)))
chain_weight_all=[0 for i in range(len(chain_all))]
chain_neg_weight_all=[0 for i in range(len(chain_all))]
chain_self_weight_all=[0 for i in range(len(chain_all))]
sample_true_talbe=[]
assert len(chain_weight_all)==len(chain_all)
case_num=0
for filename in tqdm(os.listdir(directory)):
    # check if the file is a JSON file
    if filename.endswith('.json'):
        fullpath = os.path.join(directory, filename)
        # open the JSON file
        
        true_table=[0 for i in range(len(chain_all))]
        with open(fullpath) as f:
            data = json.load(f)
            input_text=filename.split('.json')[0]
            token_length=len(data)
            if token_length<3:
                continue
            inputs = tokenizer(input_text, return_tensors="pt")
            
            case_num+=1    
            
            
            r_token=1
            s_token=0
            token_circuit_record_1token=data[s_token]
            refined_matrix_1token=torch.zeros((circuit_num,circuit_num))
            for cn in range(circuit_layer,circuit_num):
                circuit_one_listtype=token_circuit_record_1token[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                for layer in range(cn//circuit_layer):
                    reverse_circuit=1-circuit_one_satisfiability[layer]
                    circuit_one_listtype[layer*29:(layer+1)*29]=reverse_circuit
                
                refined_matrix_1token[cn]=torch.IntTensor(circuit_one_listtype)
                
            token_circuit_record_selftoken=data[-1]
            refined_matrix_selftoken=torch.zeros((circuit_num,circuit_num))
            for cn in range(circuit_layer,circuit_num):
                circuit_one_listtype=token_circuit_record_selftoken[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                for layer in range(cn//circuit_layer):
                    reverse_circuit=1-circuit_one_satisfiability[layer]
                    circuit_one_listtype[layer*29:(layer+1)*29]=reverse_circuit
                
                refined_matrix_selftoken[cn]=torch.IntTensor(circuit_one_listtype)
                
            token_circuit_record_2token=data[r_token]
            refined_matrix_2token=torch.zeros((circuit_num,circuit_num))
            for cn in range(circuit_layer,circuit_num):
                circuit_one_listtype=token_circuit_record_2token[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                for layer in range(cn//circuit_layer):
                    reverse_circuit=1-circuit_one_satisfiability[layer]
                    circuit_one_listtype[layer*29:(layer+1)*29]=reverse_circuit
                
                refined_matrix_2token[cn]=torch.IntTensor(circuit_one_listtype)
                
            
            attention_flag_metric_2token=attention_flag(inputs,refined_matrix_2token,s_token,r_token)
            attention_flag_metric_2token=attention_flag_metric_2token.cpu()
            attention_flag_metric_2token=attention_flag_metric_2token.view(-1).unsqueeze(-1)
            attention_flag_metric_2token=attention_flag_metric_2token.repeat(1,circuit_num)
            attention_flag_metric_2token=torch.where(attention_flag_metric_default>0,attention_flag_metric_2token,attention_flag_metric_default)
            refined_matrix_2token=torch.where(attention_flag_metric_2token>0,refined_matrix_2token,attention_flag_metric_2token)
            
            attention_flag_metric_1token=attention_flag_metric_default
            refined_matrix_1token=torch.where(attention_flag_metric_1token>0,refined_matrix_1token,attention_flag_metric_1token)
            refined_matrix_selftoken=torch.where(attention_flag_metric_1token>0,refined_matrix_selftoken,attention_flag_metric_1token)
            
            for chain in range(len(chain_all)):
                if len(chain_all[chain])==2:
                    if refined_matrix_2token[chain_all[chain][1],chain_all[chain][0]]==1:
                        chain_weight_all[chain]+=1
                        if refined_matrix_1token[chain_all[chain][1],chain_all[chain][0]].item()!=1:
                            true_table[chain]=1
                        chain_neg_weight_all[chain]+=refined_matrix_1token[chain_all[chain][1],chain_all[chain][0]].item()
                        chain_self_weight_all[chain]+=refined_matrix_selftoken[chain_all[chain][1],chain_all[chain][0]].item()
                if len(chain_all[chain])>2:
                    chain_jump=len(chain_all[chain])
                    for jump in range(1,chain_jump):
                        
                        if refined_matrix_2token[chain_all[chain][jump],chain_all[chain][jump-1]]==0:
                            break 
                        if jump==chain_jump-1:
                            chain_weight_all[chain]+=1
                            true_table[chain]=1
                            neg_weight=0
                            self_weight=0
                            for jump_neg in range(1,chain_jump):
                                if refined_matrix_1token[chain_all[chain][jump],chain_all[chain][jump-1]]==1:
                                    neg_weight+=1
                                else:
                                    break
                            for jump_neg in range(1,chain_jump):
                                if refined_matrix_selftoken[chain_all[chain][jump],chain_all[chain][jump-1]]==1:
                                    self_weight+=1
                                else:
                                    break
                            if neg_weight==chain_jump-1:
                                chain_neg_weight_all[chain]+=1
                                true_table[chain]=0
                            if self_weight==chain_jump-1:
                                chain_self_weight_all[chain]+=1
                                true_table[chain]=0
        sample_true_talbe.append(true_table)                            
chain_weight_all=[x/case_num for x in chain_weight_all]
chain_neg_weight_all=[x/case_num for x in chain_neg_weight_all]
chain_self_weight_all=[x/case_num for x in chain_self_weight_all]
refined_matrix_skill=torch.zeros(circuit_num,circuit_num)
for chain in range(len(chain_all)):
    print_text=''
    for jump in range(len(chain_all[chain])):
            node_text='layer {} circuit {}, '.format(chain_all[chain][jump]//29,chain_all[chain][jump]%29)
            print_text=print_text+node_text
    if chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain]>0.6:
        refined_matrix_skill[chain_all[chain][1],chain_all[chain][0]]=1
        print('The chain is ',print_text, 'with positive weight {}, negative weight {}, self weight {} and pure weight {}'.format(chain_weight_all[chain],chain_neg_weight_all[chain],chain_self_weight_all[chain],chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain]))        
torch.save(refined_matrix_skill,'tensor_path/previous_token_sro_75_0.7.pt')                    
                        
case_num=0
num_2t=0
num_1t=0
for filename in tqdm(os.listdir(directory)):
    # check if the file is a JSON file
    if filename.endswith('.json'):
        fullpath = os.path.join(directory, filename)
        # open the JSON file
        
        with open(fullpath) as f:
            data = json.load(f)
            input_text=filename.split('.json')[0]
            token_length=len(data)
            if token_length<3:
                continue
            inputs = tokenizer(input_text, return_tensors="pt")   
            input_ids_ori=copy.deepcopy(inputs['input_ids'])
            attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
            inputs_1t=copy.deepcopy(inputs)
            inputs_1t['input_ids']=input_ids_ori[:,-1].unsqueeze(0)
            inputs_1t['attention_mask']=attention_mask_ori[:,-1]
            with torch.no_grad():
                outputs_2t = orig_model(**inputs, labels=inputs["input_ids"])
                final_tensor_2t=outputs_2t.logits[0][-1]
                _,label_ids_2t=torch.topk(outputs_2t.logits[0][-1],1)
                _,label_list_2t=torch.topk(outputs_2t.logits[0][-1],5)
                
                torch.cuda.empty_cache()
                outputs_1t = orig_model(**inputs_1t, labels=inputs_1t["input_ids"])
                final_tensor_1t=outputs_1t.logits[0][-1]
                _,label_ids_1t=torch.topk(outputs_1t.logits[0][-1],1)
                _,label_list_1t=torch.topk(outputs_1t.logits[0][-1],5)
                
                torch.cuda.empty_cache()
                final_tensor_cut=cut_model(inputs,refined_matrix_skill)
                _,label_ids_cut=torch.topk(final_tensor_cut,1)
                _,label_list_cut=torch.topk(final_tensor_cut,5)
                
                torch.cuda.empty_cache()
                if tokenizer.decode(label_ids_2t)==tokenizer.decode(label_ids_cut):
                    num_2t+=1
                if tokenizer.decode(label_ids_1t)==tokenizer.decode(label_ids_cut):
                    num_1t+=1
                    
                label_ids_all=torch.cat((label_list_2t,label_list_1t,label_list_cut.cpu()),dim=-1)
                rank_2t=get_token_rank(final_tensor_2t,label_ids_all)
                rank_1t=get_token_rank(final_tensor_1t,label_ids_all)
                rank_cut=get_token_rank(final_tensor_cut,label_ids_all)
                
                print(tokenizer.decode(label_ids_2t),tokenizer.decode(label_ids_1t),tokenizer.decode(label_ids_cut))
            # if case_num==0:
            #     final_tensor_2t_all=final_tensor_2t.unsqueeze(0)
            #     final_tensor_1t_all=final_tensor_1t.unsqueeze(0)
            #     final_tensor_cut_all=final_tensor_cut.unsqueeze(0)
            # else:
            #     final_tensor_2t_all=torch.cat((final_tensor_2t_all,final_tensor_2t.unsqueeze(0)),dim=0)
            #     final_tensor_1t_all=torch.cat((final_tensor_1t_all,final_tensor_1t.unsqueeze(0)),dim=0)
            #     final_tensor_cut_all=torch.cat((final_tensor_cut_all,final_tensor_cut.unsqueeze(0)),dim=0)
                
            # if case_num==0:
            #     final_tensor_2t_all=label_list_2t.unsqueeze(0)
            #     final_tensor_1t_all=label_list_1t.unsqueeze(0)
            #     final_tensor_cut_all=label_list_cut.unsqueeze(0)
            # else:
            #     final_tensor_2t_all=torch.cat((final_tensor_2t_all,label_list_2t.unsqueeze(0)),dim=0)
            #     final_tensor_1t_all=torch.cat((final_tensor_1t_all,label_list_1t.unsqueeze(0)),dim=0)
            #     final_tensor_cut_all=torch.cat((final_tensor_cut_all,label_list_cut.unsqueeze(0)),dim=0)
            if case_num==0:
                final_tensor_2t_all=rank_2t.unsqueeze(0)
                final_tensor_1t_all=rank_1t.unsqueeze(0)
                final_tensor_cut_all=rank_cut.unsqueeze(0)
                
            else:
                final_tensor_2t_all=torch.cat((final_tensor_2t_all,rank_2t.unsqueeze(0)),dim=0)
                final_tensor_1t_all=torch.cat((final_tensor_1t_all,rank_1t.unsqueeze(0)),dim=0)
                final_tensor_cut_all=torch.cat((final_tensor_cut_all,rank_cut.unsqueeze(0)),dim=0) 
                
            
              
            case_num+=1
                
            
            
            
                    
print('Rate of two tokens is {}, of one tokens is {}'.format(num_2t/case_num,num_1t/case_num))
final_tensor_2t_all=final_tensor_2t_all.cpu().numpy()
final_tensor_1t_all=final_tensor_1t_all.cpu().numpy()
final_tensor_cut_all=final_tensor_cut_all.cpu().numpy()
                    


tensors = np.concatenate((final_tensor_2t_all, final_tensor_1t_all, final_tensor_cut_all), axis=0)
tsne = TSNE(n_components=2,max_iter=800).fit_transform(tensors)


tsne_a = tsne[:case_num, :]
tsne_b = tsne[case_num:2*case_num, :]
tsne_c = tsne[2*case_num:, :]


colors = {'a': 'r', 'b': 'g', 'c': 'b'}


plt.figure(figsize=(6, 6))
plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color=colors['a'])
plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color=colors['b'])
plt.scatter(tsne_c[:, 0], tsne_c[:, 1], color=colors['c'])
plt.show()
plt.savefig('paper_figure/t_sne.jpg')

