import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_model import assert_model,refined_explain_model,ioi_check_model
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

if args.task_name=='satisfiability_explain':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        check_model=assert_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        explain_model=refined_explain_model(args)
        ioi_explain_model=ioi_check_model(args)
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
        
            
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
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
                token_length=inputs['input_ids'].size()[-1]
                with open ('json_logs/satisfiability/gpt2xl/'+args.case_type+'/'+input_case+'.json','r') as file:
                    case=json.load(file)
                refined_matrix=torch.zeros((circuit_num,circuit_num))
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_case+'_logging.log')  
                logger.info('############ CASE TEXT is'+input_case)
                logger.info('############ CASE Prediction is {}'.format(tokenizer.decode(label_ids)))
                logger.info('############ Refined Forward Graph')
                logger.info('****** Layer 1')
                for cn in range(circuit_layer,circuit_num):
                    circuit_one_listtype=case[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                    circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                    refined_matrix[cn]=torch.IntTensor(circuit_one_listtype)
                    cn_layer=cn//circuit_layer
                    logger.info('Layer {} and circuit {}'.format(cn_layer,cn%circuit_layer))
                    for cnl in range(cn_layer):
                        reserve_list=[]
                        for cinx in range(29):
                            if circuit_one_satisfiability[cnl][cinx].item()==0:
                                reserve_list.append(circuit_name[cinx])
                        logger.info('for Layer {}, the reserve circuits are {}'.format(cnl,reserve_list))
                        
                
                #check the final label is top1
                check_model(inputs,label_ids,refined_matrix)
                
                #get the circuit logits rank
                explain_model(inputs,label_ids,refined_matrix,logger)
                logging.shutdown() 
                
                
                
if args.task_name=='task_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        check_model=assert_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        explain_model=refined_explain_model(args)
        ioi_explain_model=ioi_check_model(args)
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
                
        
        
        to_be_used_matrix_all=torch.zeros((12,29))
        to_be_used_amount=torch.ones((12,1))
        for l in range(12):
            if l!=11:
                to_be_used_amount[l]=29*(11-l)
        to_be_used_amount=to_be_used_amount.repeat(1,29)
        use_matrix_all=torch.zeros((12,29))
        use_amount=torch.ones((12,1))
        for l in range(12):
            if l!=0:
                use_amount[l]=29*(l)
        use_amount=use_amount.repeat(1,29)
        effective_matrix_all=torch.zeros((12,29))   
        record=[] 
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
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
            if i==181:
                break
            
            
    
            to_be_used_matrix=torch.zeros((12,29))
            use_matrix=torch.zeros((12,29))
            effective_matrix=torch.zeros((12,29))
            with open ('json_logs/satisfiability/gpt2xl/'+args.case_type+'/'+input_case+'.json','r') as file:
                case=json.load(file)
                
            for cn in range(circuit_layer,circuit_num):
                    circuit_one_listtype=case[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                    circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype)
                    use_amount=(cn//circuit_layer)*29
                    delete_number=torch.sum(circuit_one_satisfiability)
                    use_matrix[cn//circuit_layer][cn%circuit_layer]=use_amount-delete_number
                    for c in range(use_amount):
                        if circuit_one_satisfiability[c].item()==0:
                            to_be_used_matrix[c//circuit_layer,c%circuit_layer]+=1
            effective_matrix=to_be_used_matrix+use_matrix
            effective_matrix=effective_matrix/(29*11)
            effective_matrix_all=effective_matrix_all+effective_matrix
            use_matrix_all=use_matrix_all+use_matrix
            to_be_used_matrix_all=to_be_used_matrix_all+to_be_used_matrix
            
        effective_matrix_all=effective_matrix_all/181
        use_matrix_all=use_matrix_all/181
        use_matrix_all=use_matrix_all/use_amount
        to_be_used_matrix_all=to_be_used_matrix_all/181
        to_be_used_matrix_all=to_be_used_matrix_all/to_be_used_amount
        record.append(effective_matrix_all.tolist())
        record.append(use_matrix_all.tolist())
        record.append(to_be_used_matrix_all.tolist())
        for i in range(12):
            attention=effective_matrix_all[i][1]+effective_matrix_all[i][2]+effective_matrix_all[i][3]+effective_matrix_all[i][4]+\
                effective_matrix_all[i][5]+effective_matrix_all[i][6]+effective_matrix_all[i][7]+effective_matrix_all[i][8]+\
                    effective_matrix_all[i][9]+effective_matrix_all[i][10]+effective_matrix_all[i][11]+effective_matrix_all[i][12]
            attention=attention/12
            mlp=effective_matrix_all[i][13]
            attention_mlp=effective_matrix_all[i][14]+effective_matrix_all[i][15]+effective_matrix_all[i][16]+effective_matrix_all[i][17]+\
                effective_matrix_all[i][18]+effective_matrix_all[i][19]+effective_matrix_all[i][20]+effective_matrix_all[i][21]+\
                    effective_matrix_all[i][22]+effective_matrix_all[i][23]+effective_matrix_all[i][24]+effective_matrix_all[i][25]
            attention_mlp=attention_mlp/12
            print(attention,mlp,attention_mlp)
        with open('json_logs/task_satisfiability/'+args.case_type+'.json','w',encoding='utf-8') as data:
                json.dump(record,data,ensure_ascii=False,sort_keys=True)
                
                
                
                
if args.task_name=='ioi_satisfiability':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        check_model=assert_model(args)
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        explain_model=refined_explain_model(args)
        ioi_explain_model=ioi_check_model(args)
        layer=12
        circuit_layer=29
        circuit_num=12*29
        circuit_name=['circuit0','circuit1','circuit2','circuit3','circuit4','circuit5','circuit6','circuit7','circuit8','circuit9','circuit10',\
            'circuit11','circuit12','circuit13','circuit14','circuit15','circuit16','circuit17','circuit18','circuit19','circuit20','circuit21',\
                'circuit22','circuit23','circuit24','circuit25','circuit26','circuit27','circuit28']
        
        if args.case_type=='ioidataset':
            with open('dataset/ioidataset.json','r') as f: 
                input_text=json.load(f)
                
        
        
        to_be_used_matrix_all=torch.zeros((12,29))
        to_be_used_amount=torch.ones((12,1))
        for l in range(12):
            if l!=11:
                to_be_used_amount[l]=29*(11-l)
        to_be_used_amount=to_be_used_amount.repeat(1,29)
        use_matrix_all=torch.zeros((12,29))
        use_amount=torch.ones((12,1))
        for l in range(12):
            if l!=0:
                use_amount[l]=29*(l)
        use_amount=use_amount.repeat(1,29)
        effective_matrix_all=torch.zeros((12,29))   
        record=[] 
        duplicate_weight_all=torch.zeros((12,12))
        induction_weight_all=torch.zeros((12,12))
        previous_weight_all=torch.zeros((12,12))
        Name_weight_all=torch.zeros((12,12))
        induction_weight2_all=torch.zeros((12,12))
        previous_weight2_all=torch.zeros((12,12))
        Name_weight2_all=torch.zeros((12,12))
        for i in range (len(input_text)):
            if args.case_type=='srodataset':
                input_case=input_text[i]['prompt']
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
            if i==82:
                break
            
            
    
            to_be_used_matrix=torch.zeros((12,29))
            use_matrix=torch.zeros((12,29))
            effective_matrix=torch.zeros((12,29))
            with torch.no_grad():
                outputs = orig_model(**inputs, labels=inputs["input_ids"])
                _,label_ids=torch.topk(outputs.logits[0][-1],1)
                token_length=inputs['input_ids'].size()[-1]
                
                with open ('json_logs/satisfiability/gpt2xl/'+args.case_type+'/'+input_case+'.json','r') as file:
                    case=json.load(file)
                    
                refined_matrix=torch.zeros((circuit_num,circuit_num))
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_case+'_logging.log')  
                logger.info('############ CASE TEXT is'+input_case)
                logger.info('############ CASE Prediction is {}'.format(tokenizer.decode(label_ids)))
                logger.info('############ Refined Forward Graph')
                logger.info('****** Layer 1')
                for cn in range(circuit_layer,circuit_num):
                    circuit_one_listtype=case[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                    circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype).split(29,dim=-1)
                    refined_matrix[cn]=torch.IntTensor(circuit_one_listtype)
                    cn_layer=cn//circuit_layer
                    logger.info('Layer {} and circuit {}'.format(cn_layer,cn%circuit_layer))
                    for cnl in range(cn_layer):
                        reserve_list=[]
                        for cinx in range(29):
                            if circuit_one_satisfiability[cnl][cinx].item()==0:
                                reserve_list.append(circuit_name[cinx])
                        logger.info('for Layer {}, the reserve circuits are {}'.format(cnl,reserve_list))
                #check the final label is top1
                check_model(inputs,label_ids,refined_matrix)
                
                #get the circuit logits rank
                duplicate_weight,induction_weight,induction_weight2,previous_weight,previous_weight2,Name_weight,Name_weight2=ioi_explain_model(inputs,label_ids,refined_matrix,logger,IO,IO_m1,IO_a1,S,S_m1,S_a1,S2,end)
                duplicate_weight_all=duplicate_weight_all+duplicate_weight
                induction_weight_all=induction_weight_all+induction_weight
                previous_weight_all=previous_weight_all+previous_weight
                Name_weight_all=Name_weight_all+Name_weight
                induction_weight2_all=induction_weight2_all+induction_weight2
                previous_weight2_all=previous_weight2_all+previous_weight2
                Name_weight2_all=Name_weight2_all+Name_weight2
                logging.shutdown()
                    
                for cn in range(circuit_layer,circuit_num):
                        circuit_one_listtype=case[cn-29]['layer {} and circuit {}'.format(cn//circuit_layer,cn%circuit_layer)]
                        circuit_one_satisfiability=torch.IntTensor(circuit_one_listtype)
                        use_amount=(cn//circuit_layer)*29
                        delete_number=torch.sum(circuit_one_satisfiability)
                        use_matrix[cn//circuit_layer][cn%circuit_layer]=use_amount-delete_number
                        for c in range(use_amount):
                            if circuit_one_satisfiability[c].item()==0:
                                to_be_used_matrix[c//circuit_layer,c%circuit_layer]+=1
                effective_matrix=to_be_used_matrix+use_matrix
                effective_matrix=effective_matrix/(29*11)
                effective_matrix_all=effective_matrix_all+effective_matrix
                use_matrix_all=use_matrix_all+use_matrix
                to_be_used_matrix_all=to_be_used_matrix_all+to_be_used_matrix
                
        duplicate_weight_all=duplicate_weight_all/82
        induction_weight_all=induction_weight_all/82
        previous_weight_all=previous_weight_all/82
        Name_weight_all=Name_weight_all/82
        induction_weight2_all=induction_weight2_all/82
        previous_weight2_all=previous_weight2_all/82
        Name_weight2_all=Name_weight2_all/82
        logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
        logger.info('The duplicate_weight matrix is {}'.format(duplicate_weight_all))
        logger.info('The induction_weight matrix is {}'.format(induction_weight_all))
        logger.info('The previous_weight matrix is {}'.format(previous_weight_all))
        logger.info('The name_weight matrix is {}'.format(Name_weight_all))    
        logger.info('The induction_weight2 matrix is {}'.format(induction_weight2_all))
        logger.info('The previous_weight2 matrix is {}'.format(previous_weight2_all))
        logger.info('The name_weight2 matrix is {}'.format(Name_weight2_all))
        effective_matrix_all=effective_matrix_all/82
        use_matrix_all=use_matrix_all/82
        use_matrix_all=use_matrix_all/use_amount
        to_be_used_matrix_all=to_be_used_matrix_all/82
        to_be_used_matrix_all=to_be_used_matrix_all/to_be_used_amount
        record.append(effective_matrix_all.tolist())
        record.append(use_matrix_all.tolist())
        record.append(to_be_used_matrix_all.tolist())
        for i in range(12):
            attention=effective_matrix_all[i][1]+effective_matrix_all[i][2]+effective_matrix_all[i][3]+effective_matrix_all[i][4]+\
                effective_matrix_all[i][5]+effective_matrix_all[i][6]+effective_matrix_all[i][7]+effective_matrix_all[i][8]+\
                    effective_matrix_all[i][9]+effective_matrix_all[i][10]+effective_matrix_all[i][11]+effective_matrix_all[i][12]
            attention=attention/12
            mlp=effective_matrix_all[i][13]
            attention_mlp=effective_matrix_all[i][14]+effective_matrix_all[i][15]+effective_matrix_all[i][16]+effective_matrix_all[i][17]+\
                effective_matrix_all[i][18]+effective_matrix_all[i][19]+effective_matrix_all[i][20]+effective_matrix_all[i][21]+\
                    effective_matrix_all[i][22]+effective_matrix_all[i][23]+effective_matrix_all[i][24]+effective_matrix_all[i][25]
            attention_mlp=attention_mlp/12
            print(attention,mlp,attention_mlp)
        with open('json_logs/task_satisfiability/'+args.case_type+'.json','w',encoding='utf-8') as data:
                json.dump(record,data,ensure_ascii=False,sort_keys=True)
        
        
        