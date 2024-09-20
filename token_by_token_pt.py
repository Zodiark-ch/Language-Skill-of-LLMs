import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import trunk_model,assert_model
import logging
import json
from tqdm import tqdm
import copy
from demo_representation_vocb import assert_circuits_equal_output


hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]
torch.cuda.empty_cache()
set_gpu(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

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

if args.task_name=='token_by_token':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model=trunk_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        #check_model=assert_model(args)
        #equal_model=assert_circuits_equal_output(args)
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
        if args.case_type=='orca1wc': 
            with open('dataset/OpenOrca1wordcorrect.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orca1wm': 
            with open('dataset/OpenOrca1wordmixture.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orca2wc': 
            with open('dataset/OpenOrca2wordcorrect.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orca2wm': 
            with open('dataset/OpenOrca2wordmixture.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orcaduplicate':
            with open('dataset/OpenOrcaduplicate.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='orcainductive':
            with open('dataset/OpenOrcainductive.json','r') as f: 
                input_text=json.load(f)
        
            
        for i in range (0,len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
                #input_case='Vietnam belongs to the continent of'
            elif args.case_type=='ioidataset':
                input_case=input_text[i]['text']
            elif args.case_type=='orcadataset':
                input_case=input_text[i]['text']
            else: 
                input_case=input_text[i]['text']
            print('To record {}-th case'.format(i))
            inputs = tokenizer(input_case, return_tensors="pt")
            input_ids_ori=copy.deepcopy(inputs['input_ids'])
            attention_mask_ori=copy.deepcopy(inputs['attention_mask'])
            token_length=input_ids_ori.size()[-1]
            
            for t in range(token_length-2,token_length+1):
                if t<token_length:
                    inputs['input_ids']=input_ids_ori[:,:t+1]
                    inputs['attention_mask']=attention_mask_ori[:,:t+1]
                if t==token_length:
                    inputs['input_ids']=input_ids_ori[:,t-1].unsqueeze(0)
                    inputs['attention_mask']=attention_mask_ori[:,t-1]
                with torch.no_grad():
                    
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    
                    branch_cut=torch.zeros((circuit_num,circuit_num))
                    if t==token_length-2:
                        inputs['input_ids']=torch.cat((inputs['input_ids'],label_ids.unsqueeze(0)),dim=-1)
                        inputs['attention_mask']=attention_mask_ori[:,:t+2]
                        outputs = orig_model(**inputs, labels=inputs["input_ids"])
                        _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    #init the input matrix 
                    token_num=inputs['input_ids'].size()[-1]
                    input_matrix=torch.zeros((12,29,token_num,768)).to(args.device)
                    cut_circuit_tensor_all=None
                    #equal_model(inputs)
                    top_token,input_matrix,cut_circuit_tensor_all=model(inputs,label_ids,0,0,input_matrix,cut_circuit_tensor_all)
                    assert top_token[0].item()==label_ids.item()
                    for m in tqdm(range(circuit_num)):
                        for n in range(circuit_num):
                            if m//circuit_layer > n//circuit_layer and (m+1)%29!=27 and (m+1)%29!=28 and (m+1)%29!=0 and (m+1)%29!=1 and (n+1)%29!=27 and (n+1)%29!=28 and (n+1)%29!=0 and (n+1)%29!=1 and branch_cut[m][n]!=1:
                                branch_cut[m][n]=1
                                
                                top_token,input_matrix_new,cut_circuit_tensor_all=model(inputs,label_ids,m,n,input_matrix,cut_circuit_tensor_all)
                                if top_token[0].item()!=label_ids.item():
                                    branch_cut[m][n]=0
                                else:
                                    # if t>0:
                                    #     check_model(inputs,label_ids,branch_cut)
                                    input_matrix=input_matrix_new
                                torch.cuda.empty_cache()
                                
                
                all_branch_cut=[] 
                for id in range(29,348):   
                    all_branch_cut_dict={}          
                    all_branch_cut_dict['layer {} and circuit {}'.format(id//circuit_layer,id%circuit_layer)]=branch_cut[id].tolist()
                    all_branch_cut.append(all_branch_cut_dict)
                
                    
                if t>token_length-2:
                    with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+input_case+'.json') as read_data:
                        old_data=json.load(read_data)
                        old_data.append(all_branch_cut)
                
                else:
                    old_data=[]
                    old_data.append(all_branch_cut)
                with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+input_case+'.json','w',encoding='utf-8') as data:
                    json.dump(old_data,data,ensure_ascii=False,sort_keys=True)
                    
                    
                    
                
                    
                           
                    