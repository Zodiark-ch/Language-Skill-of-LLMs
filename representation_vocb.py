import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer,GPT2LMHeadModel
from tqdm import trange
import numpy as np


class assert_FFNandproduction_gpt2xl(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.task_name=args.task_name
        self.model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.Unembedding=self.model.lm_head.weight#[E,D]
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)


        
    
            
    def forward(self):
        inputs = self.tokenizer("When Mary and John went to the store, John gave a drink to", return_tensors="pt").to(self.device)
        outputs =self.model(**inputs, labels=inputs["input_ids"])
        orig_logits=outputs.logits[0][-1]
        head_mask = [None] * 12
        transformer_outputs = self.model.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"])
        hidden_states = transformer_outputs[0]
        UT=self.Unembedding
        U=self.Unembedding.transpose(0,1)
        new_logits=torch.mm(hidden_states[0][-1].unsqueeze(0),U).squeeze()
        print('orig_logits from FFN (lm_head linear layer without bias) is', orig_logits)
        _,predicted_indices=torch.topk(orig_logits,10)
        print('max probability token_ids are:', predicted_indices)
        print('max probability tokens are:', self.tokenizer.decode(predicted_indices))
        print('###############################################')
        print('new_logits from torch.mm is:', new_logits)
        _,predicted_indices=torch.topk(new_logits,10)
        print('max probability token_ids are:', predicted_indices)
        print('max probability tokens are:', self.tokenizer.decode(predicted_indices))
        distance=F.mse_loss(new_logits,orig_logits)
        print('Taking an example from GPT2-XL, the MSE distance of  FFN and matrix production is', distance.item())

    
    
class show_each_layer_vocb(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.model_name=args.model_name
        self.task_name=args.task_name
        self.model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.Unembedding=self.model.lm_head.weight#[E,D]
        self.layers=self.model.transformer.h
        self.device=args.device
            
    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, device):
        print(f'Model: set device to {device}')
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,inputs):
        inputs=inputs.to(self.device)
        attention_mask=inputs["attention_mask"]
        input_ids=inputs['input_ids']
        batch_size=attention_mask.size()[0]
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        head_mask = [None] * 12
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.model.transformer.wte(input_ids)
        past_length = 0
        past_key_values = tuple([None] * len(self.layers))
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if layer_past is not None:
                layer_past = tuple(past_state.to(self.device) for past_state in layer_past)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=True,
                output_attentions=False,
            )
            hidden_states = outputs[0]
            ln_hidden_state=self.model.transformer.ln_f(hidden_states)
            FFN_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            print('In {} layers output, orig_logits from FFN (lm_head linear layer without bias) is'.format(i), FFN_logits)
            _,predicted_indices=torch.topk(FFN_logits,10)
            print('In {} layer output, max probability token_ids of FFN_logits are:'.format(i), predicted_indices[0])
            print('In {} layer output, max probability tokens of FFN_logits are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            print('########################################################################')
            U=self.Unembedding.transpose(0,1)
            product_logits=torch.mm(ln_hidden_state[0][-1].unsqueeze(0),U).squeeze()
            print('In {} layers output, product_logits from FFN (lm_head linear layer without bias) is'.format(i), product_logits)
            _,predicted_indices=torch.topk(product_logits,10)
            print('In {} layer output, max probability token_ids of product_logits are:'.format(i), predicted_indices)
            print('In {} layer output, max probability tokens of product_logits are:'.format(i), self.tokenizer.decode(predicted_indices))
            distance=F.mse_loss(FFN_logits,product_logits)
            print('Taking an example from GPT2-XL, the MSE distance of  FFN and matrix production is', distance.item())
            print('########################################################################')
            
            