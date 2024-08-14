from utils import get_datasets
import json



_,dataset_orig=get_datasets()
input_text=dataset_orig.sentences
word_idx=dataset_orig.word_idx

case_list=[]
for i in range (len(input_text)):
    case={}
    input_case=input_text[i]
    input_case_list=input_case.split(' ')
    new_input=''
    for t in range(len(input_case_list)-1):
        if t==0:
            new_input=input_case_list[t]
        else:
            new_input=new_input+' '+input_case_list[t]
    case['IO']=word_idx['IO'][i].item()
    case['IO-1']=word_idx['IO-1'][i].item()
    case['IO+1']=word_idx['IO+1'][i].item()
    case['S']=word_idx['S'][i].item()
    case['S-1']=word_idx['S-1'][i].item()
    case['S+1']=word_idx['S+1'][i].item()
    case['S2']=word_idx['S2'][i].item()
    case['end']=word_idx['end'][i].item()
    case['starts']=word_idx['starts'][i].item()
    case['punct']=word_idx['punct'][i].item()
    case['text']=new_input
    case_list.append(case)
    
with open('dataset/'+'ioidataset.json','w',encoding='utf-8') as data:
    json.dump(case_list,data,ensure_ascii=False,sort_keys=True)
    