import torch
from args import DeepArgs
from utils import set_gpu 
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel
from circuit_into_ebeddingspace import attention_circuit

hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)

if args.task_name=='attention_analysis':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model=attention_circuit(args)
        input_text="The Space Needle is in downtown" 
        inputs = tokenizer(input_text, return_tensors="pt")
        
        model(inputs)