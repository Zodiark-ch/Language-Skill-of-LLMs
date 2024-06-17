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


class assert_attentionmlp_equal_output(nn.Module):
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
            circuit_input=hidden_states
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
            
            #construct residual stream
            circuit_input_ln = block.ln_1(circuit_input)
            query,key,value= block.attn.c_attn(circuit_input_ln).split(768, dim=2)#mapping into attention space
            query = block.attn._split_heads(query, 12, 64)
            key = block.attn._split_heads(key, 12, 64)
            value = block.attn._split_heads(value, 12, 64)
            attn_output, attn_weights = block.attn._attn(query, key, value, attention_mask, None)
            attn_output = block.attn._merge_heads(attn_output, 12, 64)#aggregate all heads
            attn_output = block.attn.c_proj(attn_output)#mapping into residual stream
            residual_stream = circuit_input + attn_output
            ln_residual_stream = block.ln_2(residual_stream)
            ffn_residual_stream = block.mlp(ln_residual_stream)
            # residual connection
            residual_stream = residual_stream + ffn_residual_stream
            ln_hidden_state=self.model.transformer.ln_f(residual_stream)
            stream_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            print('In {} layers output, stream_logits is'.format(i), stream_logits)
            _,predicted_indices=torch.topk(stream_logits,10)
            print('In {} layer output, max probability token_ids of stream_logits are:'.format(i), predicted_indices[0])
            print('In {} layer output, max probability tokens of stream_logits are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            distance=F.mse_loss(FFN_logits,stream_logits)
            print('Taking an example from GPT2-XL, the MSE distance of  FFN and stream is', distance.item())
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            

class assert_circuits_equal_output(nn.Module):
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
            circuit_input=hidden_states
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
            
            #construct space mapping matrix
            key_length=hidden_states.size()[-2]
            W_qkv=block.attn.c_attn.weight #R^[d,3a]=[768,2304]
            W_qkvbias=block.attn.c_attn.bias #R^[3a]=[2304]
            W_qkvbias=W_qkvbias.repeat(key_length,1)#R^[N,3a]=[14,2304]
            W_q,W_k,W_v=W_qkv.split(768, dim=1)#R^[d,a]=[768,768]
            W_qbias,W_kbias,W_vbias=W_qkvbias.split(768, dim=-1)#R^[N,a]=[14,768]
            W_mhq=self._split_heads(W_q,12,64)#R^[num_head,d,a/num_head]=[12,768,64] simply H represents num_heads
            W_mhk=self._split_heads(W_k,12,64)
            W_mhv=self._split_heads(W_v,12,64)
            W_mhqbias=self._split_heads(W_qbias,12,64)#R^[num_head,N,a/num_head]=[12,14,64]
            W_mhkbias=self._split_heads(W_kbias,12,64)
            W_mhvbias=self._split_heads(W_vbias,12,64)
            W_mhqk=torch.matmul(W_mhq,W_mhk.transpose(-1,-2))#R[H, d,d]=[12,768,768]
            W_o=block.attn.c_proj.weight#R^[a,d]=[768,768]
            W_obias=block.attn.c_proj.bias#R^[d]=[768],but in practice, we used R=[N,768]
            W_obias=W_obias.repeat(key_length,1)#R^[N,a]=[14,768]
            W_mho=self._split_heads(W_o.transpose(-1,-2),12,64).transpose(-1,-2)#because a is first dim, so need transpose, R^[H,a/H,D]=[12,64,768]
            W_mhov=torch.matmul(W_mhv,W_mho)#R^[H,d,d]=[12,768,768]
            W_mlp1=block.mlp.c_fc.weight #R^[d,m]=[768,3072]
            W_mlp1bias=block.mlp.c_fc.bias #R^[m]=[3072]
            W_mlp1bias=W_mlp1bias.repeat(key_length,1)#R^[N,m]=[14,3072]
            W_mlp2=block.mlp.c_proj.weight #R^[m,d]=[3072,768]
            W_mlp2bias=block.mlp.c_proj.bias #R^[d]=[768] 
            W_mlp2bias=W_mlp2bias.repeat(key_length,1)#R^[N,m]=[14,3072]
            W_mlp=torch.mm(W_mlp1,W_mlp2)# mlp space mapping, R^[d,d]=[768,768]
            Act_mlp=block.mlp.act #activation of mlp, and activation of attention omitted is softmax
            
            
            
            #circuit_1 is the self path, only include itself
            circuit_1=circuit_input

            
            
            
            
            #circuit_2 is the attention only path, only include attention, 
            circuit2_input_ln = block.ln_1(circuit_input)# make representation matrix get normed R^[N,d]=[14,768]
            circuit2_input_ln=circuit2_input_ln.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
                #get raw attention weight A (raw compression matrix), actually A consists of 4 items
            Output_mhqk=torch.matmul(circuit2_input_ln,W_mhqk)#X*Wqk
            Output_mhqk=torch.matmul(Output_mhqk,circuit2_input_ln.transpose(-1,-2))#X*Wqk*XT, R^[H,N,N]=[12,14,14]
            
            Output_mhqkb1=torch.matmul(W_mhqbias,W_mhk.transpose(-1,-2))#bq*WkT
            Output_mhqkb1=torch.matmul(Output_mhqkb1,circuit2_input_ln.transpose(-1,-2))#bq*WkT*XT, R[H,N,N]
            
            Output_mhqkb2=torch.matmul(circuit2_input_ln,W_mhq)#X*Wq
            Output_mhqkb2=torch.matmul(Output_mhqkb2,W_mhkbias.transpose(-1,-2))#X*Wq*bkT, R[H,N,N]
            
            Output_mhqkb3=torch.matmul(W_mhqbias,W_mhkbias.transpose(-1,-2))#bq*bkT, R[H,N,N]
            
            Output_mhqk=Output_mhqk+Output_mhqkb1+Output_mhqkb2+Output_mhqkb3
            Output_mhqk = Output_mhqk / torch.full(
                [], 64 ** 0.5, dtype=Output_mhqk.dtype, device=Output_mhqk.device)
            
                #get compression matrix 
            # if only "normal" attention layer implements causal mask
            query_length, key_length = circuit2_input_ln.size(-2), circuit2_input_ln.size(-2)
            causal_mask = torch.tril(torch.ones((key_length, key_length), dtype=torch.bool)).view(
                 1, key_length, key_length).to(self.device)
            mask_value = torch.finfo(Output_mhqk.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=Output_mhqk.dtype, device=Output_mhqk.device)
            Output_mhqk = torch.where(causal_mask, Output_mhqk.to(Output_mhqk.dtype), mask_value)
            attn_weights=nn.functional.softmax(Output_mhqk, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
                #get output of OV path (representation matrix)
            Output_mhov=torch.matmul(circuit2_input_ln,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            # value=torch.matmul(circuit2_input_ln,W_mhv)+W_mhvbias
            # attn_output=torch.matmul(attn_weights,value)
            # attn_output_mapping=torch.matmul(attn_output,W_mho)
                #get production of each head and sum of all heads
            bv_Wo=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh=torch.matmul(attn_weights,Output_mhov)+torch.matmul(attn_weights,bv_Wo)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            head1_attn,head2_attn,head3_attn,head4_attn,head5_attn,head6_attn,head7_attn,head8_attn,\
                head9_attn,head10_attn,head11_attn,head12_attn=Output_mh.split(1,dim=0)
            circuit_2=head1_attn+head2_attn+head3_attn+head4_attn+head5_attn+head6_attn+head7_attn+head8_attn+head9_attn+head10_attn+head11_attn+head12_attn+W_obias
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            Output_mlp1_act_steam=Act_mlp(Output_mlp1_all-W_mlp1bias) #activated
            circuit_stream=torch.matmul(Output_mlp1_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            circuit_Wmlp1bias=circuit_stream_all-circuit_stream
            Output_mlp1_bias=Act_mlp(W_mlp1bias) #activated
            circuit_uni_wmlp1bias=torch.matmul(Output_mlp1_bias,W_mlp2)#R^[B,N,d]=[1,14,768]
            circuit_syn_bias=circuit_Wmlp1bias-circuit_uni_wmlp1bias
            
            
            
            
            #circuit_3 is the mlp only path, 
            circuit3_input_ln = block.ln_1(circuit_input)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit4_input_ln = block.ln_2(circuit_2)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            # circuit4_input_ln=circuit4_input_ln.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
            #     #get output of OV path (representation matrix)
            # Output_mhov=torch.matmul(circuit4_input_ln,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768] 
            # Output_mhovmlp1=torch.matmul(Output_mhov,W_mlp1)#X*Wov*Wmlp1, R^[H,N,a]=[12,14,3072]
            # Output_mhovmlp1_act=Act_mlp(Output_mhovmlp1) #activated
            # Output_mhovmlp=torch.matmul(Output_mhovmlp1_act,W_mlp2)#R^[H,N,d]=[12,14,768]
            #     #get production of each head and sum of all heads
            # Output_mhattnmlp=torch.matmul(attn_weights,Output_mhovmlp) # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            # head1_am,head2_am,head3_am,head4_am,head5_am,head6_am,head7_am,head8_am,\
            #     head9_am,head10_am,head11_am,head12_am=Output_mhattnmlp.split(1,dim=0)
            # circuit_4=head1_am+head2_am+head3_am+head4_am+head5_am+head6_am+head7_am+head8_am+head9_am+head10_am+head11_am+head12_am
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            circuit_5=(circuit_stream-circuit_3-circuit_4)+circuit_syn_bias
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wmlp1 and bias in Wmlp2
            circuit_6=circuit_uni_wmlp1bias+W_mlp2bias
            
            #get circuit sum 
            circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            
            
            ln_hidden_state=self.model.transformer.ln_f(circuit_sum)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            print('In {} layers output, circuit_logits is'.format(i), circuit_logits)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability token_ids of circuit_logits are:'.format(i), predicted_indices[0])
            print('In {} layer output, max probability tokens of circuit_logits are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            distance=F.mse_loss(FFN_logits,circuit_logits)
            print('Taking an example from GPT2-XL, the MSE distance of  forward and circuit is', distance.item())
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)




class show_vocabulary_circuit(nn.Module):
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
            circuit_input=hidden_states
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
            
            #construct space mapping matrix
            key_length=hidden_states.size()[-2]
            W_qkv=block.attn.c_attn.weight #R^[d,3a]=[768,2304]
            W_qkvbias=block.attn.c_attn.bias #R^[3a]=[2304]
            W_qkvbias=W_qkvbias.repeat(key_length,1)#R^[N,3a]=[14,2304]
            W_q,W_k,W_v=W_qkv.split(768, dim=1)#R^[d,a]=[768,768]
            W_qbias,W_kbias,W_vbias=W_qkvbias.split(768, dim=-1)#R^[N,a]=[14,768]
            W_mhq=self._split_heads(W_q,12,64)#R^[num_head,d,a/num_head]=[12,768,64] simply H represents num_heads
            W_mhk=self._split_heads(W_k,12,64)
            W_mhv=self._split_heads(W_v,12,64)
            W_mhqbias=self._split_heads(W_qbias,12,64)#R^[num_head,N,a/num_head]=[12,14,64]
            W_mhkbias=self._split_heads(W_kbias,12,64)
            W_mhvbias=self._split_heads(W_vbias,12,64)
            W_mhqk=torch.matmul(W_mhq,W_mhk.transpose(-1,-2))#R[H, d,d]=[12,768,768]
            W_o=block.attn.c_proj.weight#R^[a,d]=[768,768]
            W_obias=block.attn.c_proj.bias#R^[d]=[768],but in practice, we used R=[N,768]
            W_obias=W_obias.repeat(key_length,1)#R^[N,a]=[14,768]
            W_mho=self._split_heads(W_o.transpose(-1,-2),12,64).transpose(-1,-2)#because a is first dim, so need transpose, R^[H,a/H,D]=[12,64,768]
            W_mhov=torch.matmul(W_mhv,W_mho)#R^[H,d,d]=[12,768,768]
            W_mlp1=block.mlp.c_fc.weight #R^[d,m]=[768,3072]
            W_mlp1bias=block.mlp.c_fc.bias #R^[m]=[3072]
            W_mlp1bias=W_mlp1bias.repeat(key_length,1)#R^[N,m]=[14,3072]
            W_mlp2=block.mlp.c_proj.weight #R^[m,d]=[3072,768]
            W_mlp2bias=block.mlp.c_proj.bias #R^[d]=[768] 
            W_mlp2bias=W_mlp2bias.repeat(key_length,1)#R^[N,m]=[14,3072]
            W_mlp=torch.mm(W_mlp1,W_mlp2)# mlp space mapping, R^[d,d]=[768,768]
            Act_mlp=block.mlp.act #activation of mlp, and activation of attention omitted is softmax
            
            
            
            #circuit_1 is the self path, only include itself
            circuit_1=circuit_input

            
            
            
            
            #circuit_2 is the attention only path, only include attention, 
            circuit2_input_ln = block.ln_1(circuit_input)# make representation matrix get normed R^[N,d]=[14,768]
            circuit2_input_ln=circuit2_input_ln.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
                #get raw attention weight A (raw compression matrix), actually A consists of 4 items
            Output_mhqk=torch.matmul(circuit2_input_ln,W_mhqk)#X*Wqk
            Output_mhqk=torch.matmul(Output_mhqk,circuit2_input_ln.transpose(-1,-2))#X*Wqk*XT, R^[H,N,N]=[12,14,14]
            
            Output_mhqkb1=torch.matmul(W_mhqbias,W_mhk.transpose(-1,-2))#bq*WkT
            Output_mhqkb1=torch.matmul(Output_mhqkb1,circuit2_input_ln.transpose(-1,-2))#bq*WkT*XT, R[H,N,N]
            
            Output_mhqkb2=torch.matmul(circuit2_input_ln,W_mhq)#X*Wq
            Output_mhqkb2=torch.matmul(Output_mhqkb2,W_mhkbias.transpose(-1,-2))#X*Wq*bkT, R[H,N,N]
            
            Output_mhqkb3=torch.matmul(W_mhqbias,W_mhkbias.transpose(-1,-2))#bq*bkT, R[H,N,N]
            
            Output_mhqk=Output_mhqk+Output_mhqkb1+Output_mhqkb2+Output_mhqkb3
            Output_mhqk = Output_mhqk / torch.full(
                [], 64 ** 0.5, dtype=Output_mhqk.dtype, device=Output_mhqk.device)
            
                #get compression matrix 
            # if only "normal" attention layer implements causal mask
            query_length, key_length = circuit2_input_ln.size(-2), circuit2_input_ln.size(-2)
            causal_mask = torch.tril(torch.ones((key_length, key_length), dtype=torch.bool)).view(
                 1, key_length, key_length).to(self.device)
            mask_value = torch.finfo(Output_mhqk.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=Output_mhqk.dtype, device=Output_mhqk.device)
            Output_mhqk = torch.where(causal_mask, Output_mhqk.to(Output_mhqk.dtype), mask_value)
            attn_weights=nn.functional.softmax(Output_mhqk, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
                #get output of OV path (representation matrix)
            Output_mhov=torch.matmul(circuit2_input_ln,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            # value=torch.matmul(circuit2_input_ln,W_mhv)+W_mhvbias
            # attn_output=torch.matmul(attn_weights,value)
            # attn_output_mapping=torch.matmul(attn_output,W_mho)
                #get production of each head and sum of all heads
            bv_Wo=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh=torch.matmul(attn_weights,Output_mhov)+torch.matmul(attn_weights,bv_Wo)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            head1_attn,head2_attn,head3_attn,head4_attn,head5_attn,head6_attn,head7_attn,head8_attn,\
                head9_attn,head10_attn,head11_attn,head12_attn=Output_mh.split(1,dim=0)
            circuit_2=head1_attn+head2_attn+head3_attn+head4_attn+head5_attn+head6_attn+head7_attn+head8_attn+head9_attn+head10_attn+head11_attn+head12_attn+W_obias
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            Output_mlp1_act_steam=Act_mlp(Output_mlp1_all-W_mlp1bias) #activated
            circuit_stream=torch.matmul(Output_mlp1_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            circuit_Wmlp1bias=circuit_stream_all-circuit_stream
            Output_mlp1_bias=Act_mlp(W_mlp1bias) #activated
            circuit_uni_wmlp1bias=torch.matmul(Output_mlp1_bias,W_mlp2)#R^[B,N,d]=[1,14,768]
            circuit_syn_bias=circuit_Wmlp1bias-circuit_uni_wmlp1bias
            
            
            
            
            #circuit_3 is the mlp only path, 
            circuit3_input_ln = block.ln_1(circuit_input)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit4_input_ln = block.ln_2(circuit_2)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
    
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            circuit_5=(circuit_stream-circuit_3-circuit_4)+circuit_syn_bias
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wmlp1 and bias in Wmlp2
            circuit_6=circuit_uni_wmlp1bias+W_mlp2bias
            
            #get circuit sum 
            circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            
            #show circuit_sum vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_sum)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_sum are:'.format(i), self.tokenizer.decode(predicted_indices[0])) 
            
            #show circuit_1 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_1)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_1 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            #show circuit_2 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_2)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_2 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            #show circuit_3 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_3)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_3 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            #show circuit_4 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_4)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_4 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            #show circuit_5 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_5)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_5 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            #show circuit_6 vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_6)
            circuit_logits=self.model.lm_head(ln_hidden_state)[-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            print('In {} layer output, max probability tokens of circuit_6 are:'.format(i), self.tokenizer.decode(predicted_indices[0]))
            
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
                       