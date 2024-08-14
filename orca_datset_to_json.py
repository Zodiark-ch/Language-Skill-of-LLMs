from utils import get_datasets
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2LMHeadModel,LlamaTokenizer,LlamaForCausalLM
import argparse

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model =GPT2LMHeadModel.from_pretrained("openai-community/gpt2")


ds = load_dataset("Open-Orca/OpenOrca")
data=ds['train']
data_length=len(data)
num=0
batch_squeeze=torch.ones((1,16))
sample_sequeece=np.random.choice(data_length,10000000)
sample_sequeece=sample_sequeece[:5000]
case_list=[]
check_duplicate=[]
for i in range(len(sample_sequeece)):
    
    data_id=int(sample_sequeece[i])
    question=data[data_id]['question']
    question_word=question.split(' ')
    for wl in range(min(len(question_word)-2,20)):
        case={}
        inputs = tokenizer(question_word[wl], return_tensors="pt")
        input_ids=inputs["input_ids"][0]
        inputs_a1=tokenizer(question_word[wl+1], return_tensors="pt")
        input_a1_ids=inputs_a1["input_ids"][0]
        if input_ids.size()[-1]==1 and input_a1_ids.size()[-1]==1:
            text=question_word[wl]+' '+question_word[wl+1]
            label_word = tokenizer(' '+question_word[wl+2], return_tensors="pt")
            label_word_ids=label_word["input_ids"][0][0]
            input_text=tokenizer(text, return_tensors="pt")
            outputs = model(**input_text, labels=input_text["input_ids"])
            _,label_ids=torch.topk(outputs.logits[0][-1],1)
            #if label_ids==label_word_ids:
            
            if text not in check_duplicate:
                case['text']=text
                case['answer']=tokenizer.decode(label_ids)
                check_duplicate.append(text)
                case_list.append(case)
                num=num+1
    print(num)
    if num>800:
        break
            
            
    
    
    
        
        
with open('dataset/'+'OpenOrca2wordmixture.json','w',encoding='utf-8') as data:
    json.dump(case_list,data,ensure_ascii=False,sort_keys=True)