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
        if args.case_type=='icl_sst2':
            with open('dataset/icl_sst2.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='icl_oc':
            with open('dataset/icl_object_counting.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='icl_raco':
            with open('dataset/icl_reasoning_about_colored_objects.json','r') as f: 
                input_text=json.load(f)
        if args.case_type=='icl_qawiki':
            with open('dataset/icl_qa_wikidata.json','r') as f: 
                input_text=json.load(f)
        
        input_self_text=':'
        inputs_self = tokenizer(input_self_text, return_tensors="pt")
        outputs_self = orig_model(**inputs_self, labels=inputs_self["input_ids"])
        _,self_label_ids=torch.topk(outputs_self.logits[0][-1],1)
        branch_cut=torch.zeros((circuit_num,circuit_num))
                
        #init the input matrix 
        token_num=inputs_self['input_ids'].size()[-1]
        input_matrix=torch.zeros((12,29,token_num,768)).to(args.device)
        cut_circuit_tensor_all=None
        #equal_model(inputs)
        with torch.no_grad():
            top_token,input_matrix,cut_circuit_tensor_all=model(inputs_self,self_label_ids,0,0,input_matrix,cut_circuit_tensor_all)
            assert top_token[0].item()==self_label_ids.item()
            for m in tqdm(range(circuit_num)):
                for n in range(circuit_num):
                    if m//circuit_layer > n//circuit_layer and (m+1)%29!=27 and (m+1)%29!=28 and (m+1)%29!=0 and (m+1)%29!=1 and (n+1)%29!=27 and (n+1)%29!=28 and (n+1)%29!=0 and (n+1)%29!=1 and branch_cut[m][n]!=1:
                        branch_cut[m][n]=1
                        
                        top_token,input_matrix_new,cut_circuit_tensor_all=model(inputs_self,self_label_ids,m,n,input_matrix,cut_circuit_tensor_all)
                        if top_token[0].item()!=self_label_ids.item():
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
                
        old_data=[]
        old_data.append(all_branch_cut)
        with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+'self.json','w',encoding='utf-8') as data:
            json.dump(old_data,data,ensure_ascii=False,sort_keys=True)
            
        for i in range (0,len(input_text)):
            #sst2 571
            case_idx=input_text[i]['id']
            print('To record {}-th case'.format(i))
            
            
            for t in range(0,2):
                if t==0:
                    input_case=input_text[i]['question']
                    inputs = tokenizer(input_case, return_tensors="pt")
                    token_length=inputs["input_ids"].size()[-1]
                if t==1:
                    input_case=input_text[i]['text']
                    inputs = tokenizer(input_case, return_tensors="pt")
                    token_length=inputs["input_ids"].size()[-1]
                with torch.no_grad():
                    
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,label_ids=torch.topk(outputs.logits[0][-1],1)
                    
                    branch_cut=torch.zeros((circuit_num,circuit_num))
                    
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
                
                    
                if t>0:
                    with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+'self.json') as self_data:
                        self_matrix=json.load(self_data)
                        self_cut=self_matrix[0]
                    with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+str(case_idx)+'.json') as read_data:
                        old_data=json.load(read_data)
                        old_data.append(all_branch_cut)
                        old_data.append(self_cut)
                
                else:
                    old_data=[]
                    old_data.append(all_branch_cut)
                with open('json_logs/token_by_token/gpt2xl/'+args.case_type+'/'+str(case_idx)+'.json','w',encoding='utf-8') as data:
                    json.dump(old_data,data,ensure_ascii=False,sort_keys=True)
                    
                    
                    
                
                    
                           
                    