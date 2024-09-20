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

# specify the directory you want to read files from
directory = 'json_logs/token_by_token/gpt2xl/icl_sst2'
circuit_layer=29
circuit_num=12*29    

hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]
torch.cuda.empty_cache()
set_gpu(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'


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
    logger = get_logger('logs/' +'icl_sst2_logging.log')
        
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
                    
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    #check_model(inputs,label_ids,assert_refined_matrix)
                    
                    attention_flag_metric=attention_flag_metric_default
                    refined_matrix=torch.where(attention_flag_metric>0,refined_matrix,attention_flag_metric)
                    #print(refined_matrix[29][:29])
                    
                    if t==2:
                        negative_metrix=negative_metrix+refined_matrix
                        nagetive_num+=1
                    elif t==1: 
                        positive_metrix=positive_metrix+refined_matrix
                        positive_num+=1
                    # else:
                    #     self_metrix=self_metrix+refined_matrix
                    #     self_num+=1
                            
    negative_metrix=negative_metrix/nagetive_num
    positive_metrix=positive_metrix/positive_num
    #self_metrix=self_metrix/self_num
    skill_metrix=positive_metrix-negative_metrix
    skill_metrix=torch.where(skill_metrix>0,skill_metrix,0)
    skill_topk=torch.where(skill_metrix>0.6,skill_metrix,0)
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
    
    img_pst,im_pst=plot_cka_matrix(positive_metrix)
    #text=annotate_heatmap(im_pst,valfmt="{x:.2f}")
    img_pst.savefig('paper_figure/positive.jpg')
    
    img_ngt,im_ngt=plot_cka_matrix(negative_metrix)
    #text=annotate_heatmap(im_ngt,valfmt="{x:.2f}")
    img_ngt.savefig('paper_figure/negative.jpg')
    
    img_skl,im_skl=plot_cka_matrix(skill_metrix)
    #text=annotate_heatmap(im_skl,valfmt="{x:.2f}")
    img_skl.savefig('paper_figure/skill_metrix.jpg')
    
    chain_weight_all=[0 for i in range(len(chain_all))]
    chain_neg_weight_all=[0 for i in range(len(chain_all))]
    chain_self_weight_all=[0 for i in range(len(chain_all))]
    sample_true_talbe=[]
    assert len(chain_weight_all)==len(chain_all)
 
    real_case_num=0
    for filename in tqdm(os.listdir(directory)):
        # check if the file is a JSON file
        if filename.endswith('.json'):
            fullpath = os.path.join(directory, filename)
            # open the JSON file
            true_table=[0 for i in range(len(chain_all))]
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
            real_case_num+=1    
            
            token_circuit_record_1token=data[0]
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
                
            token_circuit_record_2token=data[1]
            refined_matrix_2token=torch.zeros((circuit_num,circuit_num))
            for cn in range(circuit_layer,circuit_num):
                circuit_one_listtype=token_circuit_record_2token[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                for layer in range(cn//circuit_layer):
                    reverse_circuit=1-circuit_one_satisfiability[layer]
                    circuit_one_listtype[layer*29:(layer+1)*29]=reverse_circuit
                
                refined_matrix_2token[cn]=torch.IntTensor(circuit_one_listtype)
                
            
            
            attention_flag_metric_1token=attention_flag_metric_default
            refined_matrix_1token=torch.where(attention_flag_metric_1token>0,refined_matrix_1token,attention_flag_metric_1token)
            refined_matrix_selftoken=torch.where(attention_flag_metric_1token>0,refined_matrix_selftoken,attention_flag_metric_1token)
            refined_matrix_2token=torch.where(attention_flag_metric_1token>0,refined_matrix_2token,attention_flag_metric_1token)
            
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
            sample_true_talbe.append(true_table)                            
    chain_weight_all=[x/real_case_num for x in chain_weight_all]
    chain_neg_weight_all=[x/real_case_num for x in chain_neg_weight_all]
    chain_self_weight_all=[x/real_case_num for x in chain_self_weight_all]
    chain_recoder=[]
    for chain in range(len(chain_all)):
        print_text=''
        for jump in range(len(chain_all[chain])):
            node_text='layer {} circuit {}, '.format(chain_all[chain][jump]//29,chain_all[chain][jump]%29)
            print_text=print_text+node_text
        if chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain]>0.7:
            print('The chain is ',print_text, 'with positive weight {}, negative weight {}, self weight {} and pure weight {}'.format(chain_weight_all[chain],chain_neg_weight_all[chain],chain_self_weight_all[chain],chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain]))        
            logger.info(print_text+ 'with effect {}'.format(round(chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain],2)))
            chain_recoder_one=chain_all[chain]
            chain_recoder_one.append(chain_weight_all[chain]-chain_neg_weight_all[chain]-chain_self_weight_all[chain])
            chain_recoder.append(chain_recoder_one)
    with open('json_logs/token_by_token/gpt2xl/'+'language_skill.json','w',encoding='utf-8') as data:
            json.dump(chain_recoder,data,ensure_ascii=False,sort_keys=True)            
                            
    
                
            
        
                        
    
    
    
    skill_metrix=skill_metrix.numpy().tolist()
    with open('json_logs/language_skill/'+'skill.json','w',encoding='utf-8') as dd:
                json.dump(skill_metrix,dd,ensure_ascii=False,sort_keys=True)
    negative_metrix=negative_metrix.numpy().tolist()
    with open('json_logs/language_skill/'+'negative.json','w',encoding='utf-8') as dd:
                json.dump(negative_metrix,dd,ensure_ascii=False,sort_keys=True)
    positive_metrix=positive_metrix.numpy().tolist()
    with open('json_logs/language_skill/'+'positive.json','w',encoding='utf-8') as dd:
                json.dump(positive_metrix,dd,ensure_ascii=False,sort_keys=True)
                    
    cluster_num=2              
    kmeans = KMeans(n_clusters = cluster_num, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
    y_kmeans = kmeans.fit_predict(sample_true_talbe)
    
    # plt.scatter(sample_true_talbe[y_kmeans == 0, 0], sample_true_talbe[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
    # plt.scatter(sample_true_talbe[y_kmeans == 1, 0], sample_true_talbe[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
    # plt.scatter(kmeans.cluster_centers_[:, 0],  kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    # plt.title('Clusters of clients')
    # plt.xlabel('Annual Income (k$)')
    # plt.ylabel('Spending Score (1-100)')
    # plt.legend()
    # plt.show()
    # plt.savefig("paper_figure/cluster2.jpg")
    print('k_means cluster list is', y_kmeans)
    
    cluster_sample_idx=[]
    for cluster in range(cluster_num):
        cluster_idx=[]
        for case in range(len(y_kmeans)):
            if y_kmeans[case]==cluster:
                cluster_idx.append(case)
        cluster_sample_idx.append(cluster_idx)
        
    #cluster_sample_idx is a [C, N*] list, C is the number of cluster, N* is the number of cases in this cluster
    chain_weight=[0 for x in range(len(chain_all))]
    cluser_chain_weight=[[] for x in range(cluster_num)]
    for case in range(len(sample_true_talbe)):
        for cluster in range(cluster_num):
            if case in cluster_sample_idx[cluster]:
                cluser_chain_weight[cluster].append(sample_true_talbe[case])
    assert len(cluser_chain_weight[0])==len(cluster_sample_idx[0])
    
    
    #get average
    # for cluster in range(cluster_num):
    #     cluser_chain_weight[cluster]=[x/len(cluster_sample_idx[cluster]) for x in cluser_chain_weight[cluster]]
    #     print('the {}-th cluster has chain weight'.format(cluster), cluser_chain_weight[cluster])
        
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {'weight' : 'normal',
    'size'   : 32,}

    fig=plt.figure(figsize=(25,10))
    iters=list(range(14))
    
    def draw_line(name_of_alg,color_index,datas):
        color=palette(color_index)
        avg=np.mean(datas,axis=0)
        std=np.std(datas,axis=0)
        r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
        r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
        
        ##################
        x_values = range(0, len(datas[0]), 1)
        plt.xticks(x_values)
        plt.plot(x_values, avg, color=color, label=name_of_alg, linewidth=3)
        plt.fill_between(x_values, r1, r2, color=color, alpha=0.2)
        ##################
        # plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3)
        # plt.fill_between(iters, r1, r2, color=color, alpha=0.2)
    
    for cluster in range(cluster_num):
        draw_line("cluser {}".format(cluster),cluster,cluser_chain_weight[cluster])
    
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=36)
    plt.xlabel('chain_idx',fontsize=10)
    plt.ylabel('rate',fontsize=50)
    plt.legend(loc='upper left',prop=font1)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    # plt.xlabel('Token',fontsize=32)
    # plt.ylabel('E(l)',fontsize=32)
    # plt.legend(loc='upper left',prop=font1)

    plt.tight_layout()
    plt.savefig('paper_figure/cluser_{}.jpg'.format(cluster_num))
    # plt.savefig('paper_figure/context_GPTXL_2.jpg')
    plt.show()    
    
    
    # real_case_num=0
    # new_directory_cluster0='json_logs/token_by_token/gpt2xl/icl_sst2_cluster0'
    # new_directory_cluster1='json_logs/token_by_token/gpt2xl/icl_sst2_cluster1'
    # if not os.path.exists(new_directory_cluster0):
    #     os.mkdir(new_directory_cluster0)
    # if not os.path.exists(new_directory_cluster1):
    #     os.mkdir(new_directory_cluster1)
    # for filename in tqdm(os.listdir(directory)):
    #     # check if the file is a JSON file
    #     if filename.endswith('.json'):
    #         fullpath = os.path.join(directory, filename)
    #         new_c0=os.path.join(new_directory_cluster0, filename)
    #         new_c1=os.path.join(new_directory_cluster1, filename)
    #         with open(fullpath) as f:
    #             data = json.load(f)
    #             circuit_length=len(data)
    #             if circuit_length!=3:
    #                 continue
    #         # open the JSON file
    #         if y_kmeans[real_case_num]==0:
    #             shutil.copy(fullpath, new_c0)
    #         if y_kmeans[real_case_num]==1:
    #             shutil.copy(fullpath, new_c1)
    #         real_case_num+=1