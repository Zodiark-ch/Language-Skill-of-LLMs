import torch,os
from args import DeepArgs
from utils import set_gpu,get_datasets,generate_figure
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_into_ebeddingspace import attention_circuit_check,ioi_attention_circuit,circuit_analysis,tokens_extraction,residual_analysis,\
    bias_analysis,attention_analysis,mlp_analysis,distribution_analysis,satisfiability_analysis
import logging
import json
from demo_representation_vocb import assert_circuits_equal_output,show_each_layer_vocb

hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

if args.task_name=='ioi_check':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='case':
            model=attention_circuit_check(args)
            input_text="The Space Needle is in downtown" 
            inputs = tokenizer(input_text, return_tensors="pt")
            model(inputs)
        
        if args.case_type=='ioidataset':
            #ioidataset provides visulization of function heads
            model=ioi_attention_circuit(args)
            _,dataset_orig=get_datasets()
            input_text=dataset_orig.sentences
            word_idx=dataset_orig.word_idx
            duplicate_weight_all=torch.zeros((12,12))
            induction_weight_all=torch.zeros((12,12))
            previous_weight_all=torch.zeros((12,12))
            Name_weight_all=torch.zeros((12,12))
            induction_weight2_all=torch.zeros((12,12))
            previous_weight2_all=torch.zeros((12,12))
            Name_weight2_all=torch.zeros((12,12))
            for i in range (len(input_text)):
                input_case=input_text[i]
                inputs = tokenizer(input_case, return_tensors="pt")
                IO=word_idx['IO'][i]
                IOm1=word_idx['IO-1'][i]
                IOa1=word_idx['IO+1'][i]
                S=word_idx['S'][i]
                Sm1=word_idx['S-1'][i]
                Sa1=word_idx['S+1'][i]
                S2=word_idx['S2'][i]
                end=word_idx['end'][i]
                with torch.no_grad():
                    duplicate_weight,induction_weight,induction_weight2,previous_weight,previous_weight2,Name_weight,Name_weight2=model(inputs,input_text[i],word_idx,IO,IOm1,IOa1,S,Sm1,Sa1,S2,end)
                duplicate_weight_all=duplicate_weight_all+duplicate_weight
                induction_weight_all=induction_weight_all+induction_weight
                previous_weight_all=previous_weight_all+previous_weight
                Name_weight_all=Name_weight_all+Name_weight
                induction_weight2_all=induction_weight2_all+induction_weight2
                previous_weight2_all=previous_weight2_all+previous_weight2
                Name_weight2_all=Name_weight2_all+Name_weight2
            duplicate_weight_all=duplicate_weight_all/500
            induction_weight_all=induction_weight_all/500
            previous_weight_all=previous_weight_all/500
            Name_weight_all=Name_weight_all/500
            induction_weight2_all=induction_weight2_all/500
            previous_weight2_all=previous_weight2_all/500
            Name_weight2_all=Name_weight2_all/500
            logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
            logger.info('The duplicate_weight matrix is {}'.format(duplicate_weight_all))
            logger.info('The induction_weight matrix is {}'.format(induction_weight_all))
            logger.info('The previous_weight matrix is {}'.format(previous_weight_all))
            logger.info('The name_weight matrix is {}'.format(Name_weight_all))    
            logger.info('The induction_weight2 matrix is {}'.format(induction_weight2_all))
            logger.info('The previous_weight2 matrix is {}'.format(previous_weight2_all))
            logger.info('The name_weight2 matrix is {}'.format(Name_weight2_all))
            

if args.task_name=='circuit_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='ioidataset':
            #ioidataset provides visulization of function heads
            model=circuit_analysis(args)
            _,dataset_orig=get_datasets()
            input_text=dataset_orig.sentences
            
            cos_matrix_all=torch.zeros(12,6)  
            top_cos_matrix_all=torch.zeros(12,6)  
            mse_matrix_all=torch.zeros(12,6)  
            top_mse_matrix_all=torch.zeros(12,6) 
            ce_matrix_all=torch.zeros(12,6)  
            top_ce_matrix_all=torch.zeros(12,6) 
            jsd_matrix_all=torch.zeros(12,6)  
            top_jsd_matrix_all=torch.zeros(12,6)  
            for i in range (len(input_text)):
                input_case=input_text[i]
                inputs = tokenizer(input_case, return_tensors="pt")
                print('To record {}-th case'.format(i))
                with torch.no_grad():
                    cos_matrix,top_cos_matrix,mse_matrix,top_mse_matrix,ce_matrix,top_ce_matrix,jsd_matrix,top_jsd_matrix=model(inputs)
                    cos_matrix_all=cos_matrix_all+cos_matrix
                    top_cos_matrix_all=top_cos_matrix_all+top_cos_matrix
                    mse_matrix_all=mse_matrix_all+mse_matrix
                    top_mse_matrix_all=top_mse_matrix_all+top_mse_matrix
                    ce_matrix_all=ce_matrix_all+ce_matrix
                    top_ce_matrix_all=top_ce_matrix_all+top_ce_matrix
                    jsd_matrix_all=jsd_matrix_all+jsd_matrix
                    top_jsd_matrix_all=top_jsd_matrix_all+top_jsd_matrix
            cos_matrix_all=cos_matrix_all/i        
            top_cos_matrix_all=top_cos_matrix_all/i
            mse_matrix_all=mse_matrix_all/i    
            top_mse_matrix_all=top_mse_matrix_all/i
            ce_matrix_all=ce_matrix_all/i    
            top_ce_matrix_all=top_ce_matrix_all/i
            jsd_matrix_all=jsd_matrix_all/i
            top_jsd_matrix_all=top_jsd_matrix_all/i
            logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
            logger.info('The cos_matrix_all matrix is {}'.format(cos_matrix_all))
            logger.info('The top_cos_matrix_all matrix is {}'.format(top_cos_matrix_all))
            logger.info('The mse_matrix_all matrix is {}'.format(mse_matrix_all))
            logger.info('The top_mse_matrix_all matrix is {}'.format(top_mse_matrix_all))
            logger.info('The ce_matrix_all matrix is {}'.format(ce_matrix_all))
            logger.info('The top_ce_matrix_all matrix is {}'.format(top_ce_matrix_all))
            logger.info('The jsd_matrix_all matrix is {}'.format(jsd_matrix_all))
            logger.info('The top_jsd_matrix_all matrix is {}'.format(top_jsd_matrix_all))
                    
        if args.case_type=='srodataset':
                   #srodataset provides visulization of circuits and traces
            model=circuit_analysis(args)
            
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            
            cos_matrix_all=torch.zeros(12,6)  
            top_cos_matrix_all=torch.zeros(12,6)  
            mse_matrix_all=torch.zeros(12,6)  
            top_mse_matrix_all=torch.zeros(12,6) 
            ce_matrix_all=torch.zeros(12,6)  
            top_ce_matrix_all=torch.zeros(12,6) 
            jsd_matrix_all=torch.zeros(12,6)  
            top_jsd_matrix_all=torch.zeros(12,6)  
            i=0
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
                with torch.no_grad():
                    cos_matrix,top_cos_matrix,mse_matrix,top_mse_matrix,ce_matrix,top_ce_matrix,jsd_matrix,top_jsd_matrix=model(inputs)
                    cos_matrix_all=cos_matrix_all+cos_matrix
                    top_cos_matrix_all=top_cos_matrix_all+top_cos_matrix
                    mse_matrix_all=mse_matrix_all+mse_matrix
                    top_mse_matrix_all=top_mse_matrix_all+top_mse_matrix
                    ce_matrix_all=ce_matrix_all+ce_matrix
                    top_ce_matrix_all=top_ce_matrix_all+top_ce_matrix
                    jsd_matrix_all=jsd_matrix_all+jsd_matrix
                    top_jsd_matrix_all=top_jsd_matrix_all+top_jsd_matrix
            cos_matrix_all=cos_matrix_all/i        
            top_cos_matrix_all=top_cos_matrix_all/i
            mse_matrix_all=mse_matrix_all/i    
            top_mse_matrix_all=top_mse_matrix_all/i
            ce_matrix_all=ce_matrix_all/i    
            top_ce_matrix_all=top_ce_matrix_all/i
            jsd_matrix_all=jsd_matrix_all/i
            top_jsd_matrix_all=top_jsd_matrix_all/i
            logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
            logger.info('The cos_matrix_all matrix is {}'.format(cos_matrix_all))
            logger.info('The top_cos_matrix_all matrix is {}'.format(top_cos_matrix_all))
            logger.info('The mse_matrix_all matrix is {}'.format(mse_matrix_all))
            logger.info('The top_mse_matrix_all matrix is {}'.format(top_mse_matrix_all))
            logger.info('The ce_matrix_all matrix is {}'.format(ce_matrix_all))
            logger.info('The top_ce_matrix_all matrix is {}'.format(top_ce_matrix_all))
            logger.info('The jsd_matrix_all matrix is {}'.format(jsd_matrix_all))
            logger.info('The top_jsd_matrix_all matrix is {}'.format(top_jsd_matrix_all))
            
            
if args.task_name=='residual_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            extraction_model=tokens_extraction(args)
            analysis_model=residual_analysis(args)
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            i=0
            initial_token_all=[]
            emerge_token_all=[]
            predicted_token_all=[]
            if args.logs=='true':
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
                with torch.no_grad():
                    top_token_matrix,top_token_alltokens,token_sequence=extraction_model(inputs)
                    if args.logs=='true':
                        logger.info('###new case###')
                        for m in range(len(token_sequence)):
                            logger.info('With source tokens ['+tokenizer.decode(inputs['input_ids'][0][:m+1])+'], predicted token with layer are: {}'.format(token_sequence[m]))
                    initial_token,emerge_token,predicted_token,initial_token_recorder,emerge_token_recorder,predicted_token_recorder=analysis_model(inputs,top_token_matrix)
                    initial_token_all.append(initial_token_recorder)
                    emerge_token_all.append(emerge_token_recorder)
                    predicted_token_all.append(predicted_token_recorder)
            generate_figure(initial_token_all,emerge_token_all,predicted_token_all)
                    
                    
                    # logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+args.case_type+'_logging.log')
                    # logger.info('The top_token matrix is {}'.format(top_token_matrix))
                    

if args.task_name=='bias_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            model=bias_analysis(args)
            
            if args.logs=='true':
                logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+'logging.log')
        
            with torch.no_grad():
                top_token_matrix,top_token,top_token_logits,top_attn_matrix,top_attn_token,top_attn_logits,top_mlp_matrix,top_mlp_token,top_mlp_logits=model()
                if args.logs=='true':
                    logger.info('###top_token_matrix is \n {}'.format(top_token_matrix))
                    logger.info('###top_token is\n {}'.format(top_token))
                    logger.info('###top_token_logits is \n {}'.format(top_token_logits))
                    logger.info('###top_attn_matrix is \n {}'.format(top_attn_matrix))
                    logger.info('###top_attn_token is\n {}'.format(top_attn_token))
                    logger.info('###top_attn_logits is \n {}'.format(top_attn_logits))
                    logger.info('###top_mlp_matrix is \n {}'.format(top_mlp_matrix))
                    logger.info('###top_mlp_token is\n {}'.format(top_mlp_token))
                    logger.info('###top_mlp_logits is \n {}'.format(top_mlp_logits))
                    
                    
if args.task_name=='attention_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            model=attention_analysis(args)
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            i=0
            
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
                # if args.logs=='true':
                #     logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_text+'_logging.log')
        
                with torch.no_grad():
                    orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,predicted_indices=torch.topk(outputs.logits[0][-1],1)
                    # if args.logs=='true':
                    #     logger.info('max probability tokens are:'+ tokenizer.decode(predicted_indices))
                    attention_weight_alllayer=model(inputs,predicted_indices)
                    

if args.task_name=='mlp_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            model=mlp_analysis(args)
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            i=0
            
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
                # if args.logs=='true':
                #     logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_text+'_logging.log')
        
                with torch.no_grad():
                    orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,predicted_indices=torch.topk(outputs.logits[0][-1],1)
                    # if args.logs=='true':
                    #     logger.info('max probability tokens are:'+ tokenizer.decode(predicted_indices))
                    model(inputs,predicted_indices)
                    
                    
                    
if args.task_name=='distribution_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            model=distribution_analysis(args)
        
            with torch.no_grad():
                model()
               
               
               
if args.task_name=='satisfiability_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if args.case_type=='srodataset':
            #assert_model=show_each_layer_vocb(args)
            model=satisfiability_analysis(args)
            with open('dataset/srodataset.json','r') as f: 
                data=json.load(f)
            i=0
            
            for case in data:
                i=i+1
                print('To record {}-th case'.format(i))
                input_text=case['prompt']
                inputs = tokenizer(input_text, return_tensors="pt")
                # if args.logs=='true':
                #     logger = get_logger('logs/' +args.task_name+'/'+ args.model_name +'/'+input_text+'_logging.log')
        
                with torch.no_grad():
                    orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
                    outputs = orig_model(**inputs, labels=inputs["input_ids"])
                    _,predicted_indices=torch.topk(outputs.logits[0][-1],1)
                    # if args.logs=='true':
                    #     logger.info('max probability tokens are:'+ tokenizer.decode(predicted_indices))
                    #assert_model(inputs)
                    model(inputs,predicted_indices)