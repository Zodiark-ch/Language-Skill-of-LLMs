import torch
from args import DeepArgs
from utils import set_gpu 
from representation_vocb import assert_FFNandproduction_gpt2xl,show_each_layer_vocb
from transformers import HfArgumentParser,AutoTokenizer,GPT2LMHeadModel


hf_parser = HfArgumentParser((DeepArgs,))
args: DeepArgs = hf_parser.parse_args_into_dataclasses()[0]

set_gpu(args.gpu)

if args.task_name=='general_discovery':
    if args.model_name=='gpt2xl':
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        input_text="When Mary and John went to the store, John gave a drink to" 
        inputs = tokenizer(input_text, return_tensors="pt")
        
        #run following functions to check the distance between FFN and matrix production
        model=assert_FFNandproduction_gpt2xl(args)
        model()
        
        #test the original output of complete LLMs
        orig_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        outputs = orig_model(**inputs, labels=inputs["input_ids"])
        print('logits is', outputs.logits, outputs.logits.size())
        _,predicted_indices=torch.topk(outputs.logits[0][-1],10)
        print('max probability token_ids are:', predicted_indices)
        print('max probability tokens are:', tokenizer.decode(predicted_indices))
        generation_output = orig_model.generate(input_ids=inputs["input_ids"], max_new_tokens=2)
        print(tokenizer.decode(generation_output[0]))
        
        #get partial output 
        model=show_each_layer_vocb(args)
        model(inputs)
        
        
        
