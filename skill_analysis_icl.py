from sklearn.manifold import TSNE
import os
import json
import torch
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model,assert_model,attention_flag_model,cut_prediction,skill_analysis,skill_analysis_cut
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
from plot import matrix_plot

# specify the directory you want to read files from
positive_directory = 'json_logs/token_by_token/gpt2xl/icl_sst2_cluster1'

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
    
def get_token_rank(representation,label_ids): 
        ranks = torch.argsort(torch.argsort(representation))
        rank_label_list=ranks[label_ids]
        rank_label_list=representation.size()[-1]-rank_label_list
        return rank_label_list    

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
attn_get=skill_analysis(args)
attn_get_cut=skill_analysis_cut(args)

attention_flag_metric_default=torch.ones(circuit_num,circuit_num)
attention_flage_metric_mask=torch.zeros(circuit_num,circuit_num)
for i in range(circuit_num):
    for j in range(circuit_num):
        if i<29 or (i+1)%29==27 or (i+1)%29==28 or (i+1)%29==0 or (i+1)%29==1 or (i+1)%29==14  or (j+1)%29==27 or (j+1)%29==28 or (j+1)%29==0 or (j+1)%29==1:
            attention_flag_metric_default[i][j]=0

    # iterate over every file in the directory
    
    
#attention weight analysis
    # positive sample: two tokens with skill path, negative sample: one token samples
attn_weight_check=torch.tensor([[2,14],[2,20],[2,22],[2,24],[3,3],[3,4],[3,5],[3,11],[3,14],[3,17],[4,3],[4,5],[5,11],[8,5],[10,10],[11,8],[11,9],[11,10],[11,11]])
x_label=attn_weight_check.numpy().tolist()
m=12
n=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

attn_matrix=[]
for n_idx in range(len(n)):
    #positive sample
    attn_weight_positive=torch.zeros_like(attn_weight_check[:,0])
    attn_weight_positive=attn_weight_positive.type(torch.FloatTensor)
    case_num=0
    for filename in tqdm(os.listdir(positive_directory)):
    # check if the file is a JSON file
        if filename.endswith('.json'):
            fullpath = os.path.join(positive_directory, filename)
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
            with torch.no_grad():
                inputs = tokenizer(input_text, return_tensors="pt")
                input_ids_ori=copy.deepcopy(inputs['input_ids'])
                inductive_token=input_ids_ori[0][-1]
                label_prompt_token=input_ids_ori[0][-2]
                label_prompt_token_previous=-3
                shot_flag=0
                first_token=0
                second_token=1
                for i in range(input_ids_ori.size()[-1]):
                    if input_ids_ori[0][i]==inductive_token and input_ids_ori[0][i-1]==label_prompt_token:
                        if shot_flag==0:
                            inductive_token_first_shot=i
                            label_prompt_token_first_shot=i-1
                            label_prompt_token_previous_first_shot=i-2
                            inductive_token_next_first_shot=i+1
                            shot_flag+=1
                            first_token_shot2=i+3
                            continue
                        if shot_flag==1:
                            inductive_token_second_shot=i
                            label_prompt_token_second_shot=i-1
                            label_prompt_token_previous_second_shot=i-2
                            inductive_token_next_second_shot=i+1
                            shot_flag+=1
                            first_token_shot3=i+3
                            break

                
                attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
                
                if n_idx==0:
                    n_token=first_token
                if n_idx==1:
                    n_token=second_token
                if n_idx==2:
                    n_token=inductive_token_first_shot
                if n_idx==3:
                    n_token=label_prompt_token_first_shot
                if n_idx==4:
                    n_token=label_prompt_token_previous_first_shot
                if n_idx==5:
                    n_token=inductive_token_next_first_shot
                if n_idx==6:
                    n_token=first_token_shot2
                if n_idx==7:
                    n_token=inductive_token_second_shot
                if n_idx==8:
                    n_token=label_prompt_token_second_shot
                if n_idx==9:
                    n_token=label_prompt_token_previous_second_shot
                if n_idx==10:
                    n_token=inductive_token_next_second_shot
                if n_idx==11:
                    n_token=first_token_shot3
                if n_idx==12:
                    n_token=label_prompt_token_previous
                if n_idx==13:
                    n_token=-2
                if n_idx==14:
                    n_token=-1
                    
                if m==0:
                    m=label_prompt_token_previous_first_shot
                    assert len(n)<4
                if m==1:
                    m=label_prompt_token_first_shot
                    assert len(n)<5
                if m==2:
                    m=inductive_token_first_shot
                    assert len(n)<6
                if m==3:
                    m=inductive_token_next_first_shot
                    assert len(n)<7
                if m==4:
                    m=first_token_shot2
                    assert len(n)<8
                if m==5:
                    m=label_prompt_token_previous_second_shot
                    assert len(n)<9
                if m==6:
                    m=label_prompt_token_second_shot
                    assert len(n)<10
                if m==7:
                    m=inductive_token_second_shot
                    assert len(n)<11
                if m==8:
                    m=inductive_token_next_second_shot
                    assert len(n)<12
                if m==9:
                    m=first_token_shot3
                    assert len(n)<13
                if m==10:
                    m=label_prompt_token_previous
                    assert len(n)<14
                if m==11:
                    m=-2
                    assert len(n)<15
                if m==12:
                    m=-1
                    assert len(n)<16
                attn_weight_2t=attn_get(inputs,attn_weight_check,m,n_token)
                attn_weight_positive=attn_weight_positive+attn_weight_2t.cpu()
                m=len(n)-3
                case_num+=1
                # inputs['input_ids']=input_ids_ori[:,-1].unsqueeze(0)
                # inputs['attention_mask']=attention_mask_ori[:,-1]
                # attn_weight_1t=attn_get(inputs,attn_weight_check,0,0)
                # attn_weight=torch.cat((attn_weight_2t.unsqueeze(0),attn_weight_1t.unsqueeze(0)),dim=0)
    attn_weight_positive=attn_weight_positive/case_num   
    row_attn_weight=attn_weight_positive.numpy()
    attn_matrix.append(row_attn_weight)
    attn_matrix=[np.round(i,2) for i in attn_matrix]
print(attn_matrix)
x_length=len(x_label)
y_length=len(n)
assert len(attn_matrix)==y_length
assert len(attn_matrix[0])==x_length
for idx in range(len(x_label)):
    x_label[idx]=str(x_label[idx])
y_label=[]
for idx in range(len(n)):
    y_label.append('token'+str(n[idx]))
save_path='paper_figure/sst2_p3b.jpg'
matrix_plot(attn_matrix,x_label,y_label,save_path)



        
        