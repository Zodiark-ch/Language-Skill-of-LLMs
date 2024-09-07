from utils import get_datasets
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,GPT2LMHeadModel,LlamaTokenizer,LlamaForCausalLM
import argparse

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model =GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

#test if the dataset is avaliable for this model
#input_text="stick gelatine Correct Order: gelatine stick\nentity featherweight Correct Order: entity featherweight\nchrysalis wallaby Correct Order:"
#input_text="The language of A Confederacy of Dunces is A:"
# inputs = tokenizer(input_text, return_tensors="pt")
# generation_output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
# print(tokenizer.decode([262]))

#sst2 1001,object_counting213,entailed_polarity28,qa_wikidata1001,fact_checker83,reasoning_about_colored_objects136


#dataset sst2
dataset = load_dataset('glue', 'sst2', split=['train', 'validation']) 





train_data=dataset[0]
data_length=len(train_data)
print(data_length)
sample_sequeece=np.random.choice(data_length,60000)

case_list=[]
check_duplicate=[]
case_num=0
sample_flag=0
for i in range(len(sample_sequeece)):
    data_id=int(sample_sequeece[i])
    data_case=train_data[data_id]
    shot0_text=data_case['sentence']
    if shot0_text in check_duplicate:
        continue
    if data_case['label']==0:
        shot0_label='negative'
    elif data_case['label']==1:
        shot0_label='positive'
    shot0=shot0_text+'Sentiment:'+" "+shot0_label+'\n'
    case={}
    if sample_flag==1:
        sample_flag=0
        continue
    for j in range(len(sample_sequeece)):
        if i==j:
            continue
        if sample_flag==1:
            break
        shot1_id=int(sample_sequeece[j])
        shot1_case=train_data[shot1_id]
        shot1_text=shot1_case['sentence']
        if shot1_text in check_duplicate:
            continue
        if shot1_case['label']==0:
            shot1_label='negative'
        elif shot1_case['label']==1:
            shot1_label='positive'
        if shot1_label==shot0_label:
            continue
        shot1=shot1_text+'Sentiment:'+" "+shot1_label+'\n'
        for q in range(len(sample_sequeece)):
            if q==i or q==j:
                continue
            question_id=int(sample_sequeece[q])
            question_case=train_data[question_id]
            question_text=question_case['sentence']
            if question_text in check_duplicate:
                continue
            if question_case['label']==0:
                question_label=' negative'
            elif question_case['label']==1:
                question_label=' positive'
            question=question_text+'Sentiment:'
            
            icl_text=shot0+shot1+question
            inputs_icl_text = tokenizer(icl_text, return_tensors="pt")
            outputs = model(**inputs_icl_text, labels=inputs_icl_text["input_ids"])
            _,label_ids=torch.topk(outputs.logits[0][-1],1)
            if tokenizer.decode(label_ids)==question_label and shot0_text not in check_duplicate and shot1_text not in check_duplicate and question_text not in check_duplicate:
                case['text']=icl_text
                case['answer']=question_label
                case['question']=question
                case['id']=case_num
                case_num+=1
                case_list.append(case)
                check_duplicate.append(shot0_text)
                check_duplicate.append(shot1_text)
                check_duplicate.append(question_text)
                sample_flag=1
                print(case_num)
                break
    if case_num>1000:
         break
                
with open('dataset/'+'icl_sst2.json','w',encoding='utf-8') as data:
     json.dump(case_list,data,ensure_ascii=False,sort_keys=True)            
    