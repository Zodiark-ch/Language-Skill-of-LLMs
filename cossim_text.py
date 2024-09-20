from sklearn.manifold import TSNE
import os
import json
import torch
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import representation_feedback
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
import torch.nn.functional as F

# specify the directory you want to read files from
positive_directory = 'json_logs/token_by_token/gpt2xl/srodataset_cluster0_cluster0'
negative_directory = 'json_logs/token_by_token/gpt2xl/srodataset_cluster0_cluster0'
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
representation_get=representation_feedback(args)


attention_flag_metric_default=torch.ones(circuit_num,circuit_num)
attention_flage_metric_mask=torch.zeros(circuit_num,circuit_num)
for i in range(circuit_num):
    for j in range(circuit_num):
        if i<29 or (i+1)%29==27 or (i+1)%29==28 or (i+1)%29==0 or (i+1)%29==1 or (i+1)%29==14  or (j+1)%29==27 or (j+1)%29==28 or (j+1)%29==0 or (j+1)%29==1:
            attention_flag_metric_default[i][j]=0

    # iterate over every file in the directory
    
    
#attention weight analysis
    # positive sample: two tokens with skill path, negative sample: one token samples
attn_weight_check=torch.tensor([[1,8],[1,18],[1,19],[1,20],[1,21],[2,1],[2,7],[2,14],[2,18],[2,20],[11,1],[11,14]])
x_label=attn_weight_check.numpy().tolist()
m=1
n=[0,1]
attn_matrix=[]

#positive sample
attn_weight_positive=torch.zeros_like(attn_weight_check[:,0])
attn_weight_positive=attn_weight_positive.type(torch.FloatTensor)
case_num=0
representation_num=attn_weight_check.size()[0]
representation_cossim=torch.zeros((representation_num,representation_num))
representation_cossim=representation_cossim.type(torch.FloatTensor)
representation_cossim_all=torch.zeros_like(representation_cossim)
for filename in tqdm(os.listdir(positive_directory)):
# check if the file is a JSON file
    if filename.endswith('.json'):
        fullpath = os.path.join(positive_directory, filename)
        # open the JSON file
        with open(fullpath) as f:
            data = json.load(f)
        input_text=filename.split('.json')[0]
        token_length=len(data)
        if token_length<3:
                continue
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids_ori=copy.deepcopy(inputs['input_ids'])
            attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
            representation_all=representation_get(inputs,attn_weight_check)
            assert representation_num==representation_all.size()[0]
            
            for i in range(representation_num):
                for j in range(representation_num):
                    cossim=F.cosine_similarity(representation_all[i].unsqueeze(0),representation_all[j].unsqueeze(0))
                    representation_cossim[i][j]=cossim[0]
            representation_cossim_all=representation_cossim_all+representation_cossim
                
            case_num+=1
            # inputs['input_ids']=input_ids_ori[:,-1].unsqueeze(0)
            # inputs['attention_mask']=attention_mask_ori[:,-1]
            # attn_weight_1t=attn_get(inputs,attn_weight_check,0,0)
            # attn_weight=torch.cat((attn_weight_2t.unsqueeze(0),attn_weight_1t.unsqueeze(0)),dim=0)
representation_cossim_all=representation_cossim_all/case_num   
print(representation_cossim_all)



        
        