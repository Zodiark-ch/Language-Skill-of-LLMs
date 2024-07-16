import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer,GPT2LMHeadModel
from tqdm import trange
import numpy as np
import logging




class trunk_model(nn.Module):
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
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,inputs,label_ids,m,n,input_matrix,cut_circuit_tensor_all):
        input_matrix_new=input_matrix.clone()
        to_cut=m
        cut_circuit=n
        to_cut_layer=m//29
        cut_circuit_layer=n//29
        
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
        
        input_ids=inputs['input_ids']
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.model.transformer.wte(input_ids)
        past_length = 0
        past_key_values = tuple([None] * len(self.layers))
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        circuit_input=hidden_states
        if cut_circuit_tensor_all is not None:
            cut_circuit_tensor=cut_circuit_tensor_all[cut_circuit%29]

        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):

            if m!=0 and to_cut_layer>i and (i!=cut_circuit_layer or n%29!=0):
                continue
            #construct space mapping matrix
            key_length=circuit_input.size()[-2]
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
            Act_mlp=block.mlp.act #activation of mlp, and activation of attention omitted is softmax
            
            
            #circuit_1 is the self path, only include itself
            if i<=to_cut_layer and m!=0:
                circuit_1_in=input_matrix[i][0].unsqueeze(0)
            else: circuit_1_in=circuit_input
            if i == to_cut_layer and to_cut%29==0 and i!=0:
                circuit_1_in=self.check_representation(circuit_1_in,cut_circuit_tensor)
            circuit_1=circuit_1_in
            if m==0  or i>=to_cut_layer :
                input_matrix_new[i][0]=circuit_1_in[0]
            

            
            
            
            
            #circuit_2 is the attention only path, only include attention, 
            if i<=to_cut_layer and m!=0:
                circuit_2_in=input_matrix[i][1:13]
            else: 
                circuit_2_in=circuit_input
                circuit_2_in=circuit_2_in.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12=circuit_2_in.split(1,dim=0)
            if i == to_cut_layer and to_cut%29==1:
                circuit_2_in_h1=self.check_representation(circuit_2_in_h1,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==2:
                circuit_2_in_h2=self.check_representation(circuit_2_in_h2,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==3:
                circuit_2_in_h3=self.check_representation(circuit_2_in_h3,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==4:
                circuit_2_in_h4=self.check_representation(circuit_2_in_h4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==5:
                circuit_2_in_h5=self.check_representation(circuit_2_in_h5,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==6:
                circuit_2_in_h6=self.check_representation(circuit_2_in_h6,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==7:
                circuit_2_in_h7=self.check_representation(circuit_2_in_h7,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==8:
                circuit_2_in_h8=self.check_representation(circuit_2_in_h8,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==9:
                circuit_2_in_h9=self.check_representation(circuit_2_in_h9,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==10:
                circuit_2_in_h10=self.check_representation(circuit_2_in_h10,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==11:
                circuit_2_in_h11=self.check_representation(circuit_2_in_h11,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==12:
                circuit_2_in_h12=self.check_representation(circuit_2_in_h12,cut_circuit_tensor)
                
            circuit_2_in=torch.cat((circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12),dim=0)
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][1:13]=circuit_2_in    
            circuit2_input_ln = block.ln_1(circuit_2_in)# make representation matrix get normed 
            
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
            _, key_length = circuit2_input_ln.size(-2), circuit2_input_ln.size(-2)
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
            
                #get production of each head and sum of all heads
            bv_Wo=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh=torch.matmul(attn_weights,Output_mhov)+torch.matmul(attn_weights,bv_Wo)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
                circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12=Output_mh.split(1,dim=0)
            circuit_2=circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2+W_obias
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #circuit_3 is the mlp only path, 
            if i<=to_cut_layer and m!=0:
                circuit_3_in=input_matrix[i][13].unsqueeze(0)
            else: 
                circuit_3_in=circuit_input
            if i == to_cut_layer and to_cut%29==13:
                circuit_3_in=self.check_representation(circuit_3_in,cut_circuit_tensor)
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][13]=circuit_3_in[0]
            circuit3_input_ln = block.ln_1(circuit_3_in)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            #new circuit_2 for circuit_4 
            if i<=to_cut_layer and m!=0:
                circuit_2_in_forc4=input_matrix[i][14:26]
            else: 
                
                circuit_2_in_forc4=circuit_input
                circuit_2_in_forc4=circuit_2_in_forc4.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
            circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4=\
                circuit_2_in_forc4.split(1,dim=0)
    
            if i == to_cut_layer and to_cut%29==14:
                circuit_2_in_h1_forc4=self.check_representation(circuit_2_in_h1_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==15:
                circuit_2_in_h2_forc4=self.check_representation(circuit_2_in_h2_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==16:
                circuit_2_in_h3_forc4=self.check_representation(circuit_2_in_h3_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==17:
                circuit_2_in_h4_forc4=self.check_representation(circuit_2_in_h4_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==18:
                circuit_2_in_h5_forc4=self.check_representation(circuit_2_in_h5_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==19:
                circuit_2_in_h6_forc4=self.check_representation(circuit_2_in_h6_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==20:
                circuit_2_in_h7_forc4=self.check_representation(circuit_2_in_h7_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==21:
                circuit_2_in_h8_forc4=self.check_representation(circuit_2_in_h8_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==22:
                circuit_2_in_h9_forc4=self.check_representation(circuit_2_in_h9_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==23:
                circuit_2_in_h10_forc4=self.check_representation(circuit_2_in_h10_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==24:
                circuit_2_in_h11_forc4=self.check_representation(circuit_2_in_h11_forc4,cut_circuit_tensor)
            if i == to_cut_layer and to_cut%29==25:
                circuit_2_in_h12_forc4=self.check_representation(circuit_2_in_h12_forc4,cut_circuit_tensor)
            
            circuit_2_in_forc4=torch.cat(( circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4),dim=0)
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][14:26]=circuit_2_in_forc4
            circuit2_input_ln_forc4 = block.ln_1(circuit_2_in_forc4)# make representation matrix get normed 
            
                #get raw attention weight A (raw compression matrix), actually A consists of 4 items
            Output_mhqk_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhqk)#X*Wqk
            Output_mhqk_forc4=torch.matmul(Output_mhqk_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#X*Wqk*XT, R^[H,N,N]=[12,14,14]
            
            Output_mhqkb1_forc4=torch.matmul(W_mhqbias,W_mhk.transpose(-1,-2))#bq*WkT
            Output_mhqkb1_forc4=torch.matmul(Output_mhqkb1_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#bq*WkT*XT, R[H,N,N]
            
            Output_mhqkb2_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhq)#X*Wq
            Output_mhqkb2_forc4=torch.matmul(Output_mhqkb2_forc4,W_mhkbias.transpose(-1,-2))#X*Wq*bkT, R[H,N,N]
            
            Output_mhqkb3_forc4=torch.matmul(W_mhqbias,W_mhkbias.transpose(-1,-2))#bq*bkT, R[H,N,N]
            
            Output_mhqk_forc4=Output_mhqk_forc4+Output_mhqkb1_forc4+Output_mhqkb2_forc4+Output_mhqkb3_forc4
            Output_mhqk_forc4 = Output_mhqk_forc4 / torch.full(
                [], 64 ** 0.5, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            
                #get compression matrix 
            # if only "normal" attention layer implements causal mask
            key_length_forc4 = circuit2_input_ln_forc4.size(-2)
            causal_mask_forc4 = torch.tril(torch.ones((key_length_forc4, key_length_forc4), dtype=torch.bool)).view(
                 1, key_length_forc4, key_length_forc4).to(self.device)
            mask_value_forc4 = torch.finfo(Output_mhqk_forc4.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value_forc4 = torch.full([], mask_value_forc4, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            Output_mhqk_forc4 = torch.where(causal_mask_forc4, Output_mhqk_forc4.to(Output_mhqk_forc4.dtype), mask_value_forc4)
            attn_weights_forc4=nn.functional.softmax(Output_mhqk_forc4, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
                #get output of OV path (representation matrix)
            Output_mhov_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            
                #get production of each head and sum of all heads
            bv_Wo_forc4=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh_forc4=torch.matmul(attn_weights_forc4,Output_mhov_forc4)+torch.matmul(attn_weights_forc4,bv_Wo_forc4)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1_forc4,circuit_2_h2_forc4,circuit_2_h3_forc4,circuit_2_h4_forc4,circuit_2_h5_forc4,circuit_2_h6_forc4,\
                circuit_2_h7_forc4,circuit_2_h8_forc4,circuit_2_h9_forc4,circuit_2_h10_forc4,circuit_2_h11_forc4,circuit_2_h12_forc4=\
                    Output_mh_forc4.split(1,dim=0)
            circuit_2_forc4=circuit_2_h1_forc4+circuit_2_h2_forc4+circuit_2_h3_forc4+circuit_2_h4_forc4+circuit_2_h5_forc4+\
                circuit_2_h6_forc4+circuit_2_h7_forc4+circuit_2_h8_forc4+circuit_2_h9_forc4+circuit_2_h10_forc4+circuit_2_h11_forc4+circuit_2_h12_forc4
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit_4_in=circuit_2_forc4+W_obias
            circuit4_input_ln = block.ln_2(circuit_4_in)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            #get subcircuit of each head and compensation circuit
            #head1
            c4h1_in=circuit_2_h1_forc4
            head1_c4in=block.ln_2(c4h1_in)
            Output_mlp1_h1=torch.matmul(head1_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h1=Act_mlp(Output_mlp1_h1) #activated
            circuit_4_h1=torch.matmul(Output_mlp1_act_4_h1,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head2
            c4h2_in=circuit_2_h2_forc4
            head2_c4in=block.ln_2(c4h2_in)
            Output_mlp1_h2=torch.matmul(head2_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h2=Act_mlp(Output_mlp1_h2) #activated
            circuit_4_h2=torch.matmul(Output_mlp1_act_4_h2,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head3 
            c4h3_in=circuit_2_h3_forc4
            head3_c4in=block.ln_2(c4h3_in)
            Output_mlp1_h3=torch.matmul(head3_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h3=Act_mlp(Output_mlp1_h3) #activated
            circuit_4_h3=torch.matmul(Output_mlp1_act_4_h3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head4
            c4h4_in=circuit_2_h4_forc4
            head4_c4in=block.ln_2(c4h4_in)
            Output_mlp1_h4=torch.matmul(head4_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h4=Act_mlp(Output_mlp1_h4) #activated
            circuit_4_h4=torch.matmul(Output_mlp1_act_4_h4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
            
            #head5
            c4h5_in=circuit_2_h5_forc4
            head5_c4in=block.ln_2(c4h5_in)
            Output_mlp1_h5=torch.matmul(head5_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h5=Act_mlp(Output_mlp1_h5) #activated
            circuit_4_h5=torch.matmul(Output_mlp1_act_4_h5,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head6
            c4h6_in=circuit_2_h6_forc4
            head6_c4in=block.ln_2(c4h6_in)
            Output_mlp1_h6=torch.matmul(head6_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h6=Act_mlp(Output_mlp1_h6) #activated
            circuit_4_h6=torch.matmul(Output_mlp1_act_4_h6,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head7
            c4h7_in=circuit_2_h7_forc4
            head7_c4in=block.ln_2(c4h7_in)
            Output_mlp1_h7=torch.matmul(head7_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h7=Act_mlp(Output_mlp1_h7) #activated
            circuit_4_h7=torch.matmul(Output_mlp1_act_4_h7,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head8
            c4h8_in=circuit_2_h8_forc4
            head8_c4in=block.ln_2(c4h8_in)
            Output_mlp1_h8=torch.matmul(head8_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h8=Act_mlp(Output_mlp1_h8) #activated
            circuit_4_h8=torch.matmul(Output_mlp1_act_4_h8,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head9
            c4h9_in=circuit_2_h9_forc4
            head9_c4in=block.ln_2(c4h9_in)
            Output_mlp1_h9=torch.matmul(head9_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h9=Act_mlp(Output_mlp1_h9) #activated
            circuit_4_h9=torch.matmul(Output_mlp1_act_4_h9,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head10
            c4h10_in=circuit_2_h10_forc4
            head10_c4in=block.ln_2(c4h10_in)
            Output_mlp1_h10=torch.matmul(head10_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h10=Act_mlp(Output_mlp1_h10) #activated
            circuit_4_h10=torch.matmul(Output_mlp1_act_4_h10,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head11
            c4h11_in=circuit_2_h11_forc4
            head11_c4in=block.ln_2(c4h11_in)
            Output_mlp1_h11=torch.matmul(head11_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h11=Act_mlp(Output_mlp1_h11) #activated
            circuit_4_h11=torch.matmul(Output_mlp1_act_4_h11,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head12
            c4h12_in=circuit_2_h12_forc4
            head12_c4in=block.ln_2(c4h12_in)
            Output_mlp1_h12=torch.matmul(head12_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h12=Act_mlp(Output_mlp1_h12) #activated
            circuit_4_h12=torch.matmul(Output_mlp1_act_4_h12,W_mlp2)#R^[B,N,d]=[1,14,768]
            
        
            
            #conpensation circuit for multi-heads, include the effects of bias in mlp1 and synergistic from interaction of multi-heads 
            if i<to_cut_layer and m!=0:
                circuit_4_compst=input_matrix[i][26].unsqueeze(0)
            else: 
                circuit_4_compst=circuit_4-circuit_4_h1-circuit_4_h2-circuit_4_h3-circuit_4_h4-circuit_4_h5-circuit_4_h6-circuit_4_h7-circuit_4_h8-\
                circuit_4_h9-circuit_4_h10-circuit_4_h11-circuit_4_h12
            
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][26]=circuit_4_compst[0]
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            if i<to_cut_layer and m!=0:
                circuit_5=input_matrix[i][27].unsqueeze(0)
            else: 
                circuit_5=(circuit_stream_all-circuit_3-circuit_4)
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][27]=circuit_5[0]
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wo,Wmlp1
            if i<to_cut_layer and m!=0:
                circuit_6=input_matrix[i][28].unsqueeze(0)
            else: 
                circuit_6=W_obias+W_mlp2bias
                circuit_6=circuit_6.unsqueeze(0)
            if m==0   or i>=to_cut_layer :
                input_matrix_new[i][28]=circuit_6[0]
            
            #get circuit sum 
            #circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            circuit_sum=circuit_1+circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+\
                circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12+circuit_3+circuit_4_h1+circuit_4_h2+circuit_4_h3+\
                    circuit_4_h4+circuit_4_h5+circuit_4_h6+circuit_4_h7+circuit_4_h8+circuit_4_h9+circuit_4_h10+circuit_4_h11+circuit_4_h12+\
                        circuit_4_compst+circuit_5+circuit_6
                        
            
            circuit_input=circuit_sum
            if i == cut_circuit_layer:
                circuit_sum_cat=torch.cat((circuit_1,circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
                circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12,circuit_3,circuit_4_h1,circuit_4_h2,circuit_4_h3,\
                    circuit_4_h4,circuit_4_h5,circuit_4_h6,circuit_4_h7,circuit_4_h8,circuit_4_h9,circuit_4_h10,circuit_4_h11,circuit_4_h12,\
                        circuit_4_compst,circuit_5,circuit_6),dim=0)#[29,N,768]
                cut_id=cut_circuit%29
                cut_circuit_tensor_all=circuit_sum_cat
                cut_circuit_tensor=cut_circuit_tensor_all[cut_id]
                    
            
        final_logits=self.get_logits(circuit_sum)
        _,predicted_indices=torch.topk(final_logits[0][-1],10)
        return predicted_indices,input_matrix_new,cut_circuit_tensor_all
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def get_softmax_logits(self,input,label_ids):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        label_logits_in=F.softmax(logits_in,dim=-1).index_select(-1,label_ids)#[1,N,1]
        
        
        return label_logits_in
    
    def get_logits(self,input):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        
        return logits_in
    
    def check_representation(self,input,cut_circuit_tensor):
        return input-cut_circuit_tensor
    
    
    
    
    
class assert_model(nn.Module):
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
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,inputs,label_ids,refined_matrix):
        # refined_matrix=[349,349]
        
        
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
        
        input_ids=inputs['input_ids']
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.model.transformer.wte(input_ids)
        past_length = 0
        past_key_values = tuple([None] * len(self.layers))
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        circuit_input=hidden_states
        key_length=circuit_input.size()[-2]
        record_circuit_tensor_all=torch.zeros((12,29,key_length,768)).to(self.device)
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):

            
            #construct space mapping matrix
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
            Act_mlp=block.mlp.act #activation of mlp, and activation of attention omitted is softmax
            
            
            #circuit_1 is the self path, only include itself
            circuit_1_in=circuit_input
            if i>0:
                check_num=i*29+0
                circuit_1_in=self.check_representation(circuit_1_in,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
            circuit_1=circuit_1_in
            
            

            
            
            
            
            #circuit_2 is the attention only path, only include attention, 
            circuit_2_in=circuit_input
            circuit_2_in=circuit_2_in.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12=circuit_2_in.split(1,dim=0)
            if i>0:
                check_num=i*29+1
                circuit_2_in_h1=self.check_representation(circuit_2_in_h1,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+2
                circuit_2_in_h2=self.check_representation(circuit_2_in_h2,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+3
                circuit_2_in_h3=self.check_representation(circuit_2_in_h3,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+4
                circuit_2_in_h4=self.check_representation(circuit_2_in_h4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+5
                circuit_2_in_h5=self.check_representation(circuit_2_in_h5,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+6
                circuit_2_in_h6=self.check_representation(circuit_2_in_h6,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+7
                circuit_2_in_h7=self.check_representation(circuit_2_in_h7,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+8
                circuit_2_in_h8=self.check_representation(circuit_2_in_h8,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+9
                circuit_2_in_h9=self.check_representation(circuit_2_in_h9,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+10
                circuit_2_in_h10=self.check_representation(circuit_2_in_h10,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+11
                circuit_2_in_h11=self.check_representation(circuit_2_in_h11,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+12
                circuit_2_in_h12=self.check_representation(circuit_2_in_h12,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                
            circuit_2_in=torch.cat((circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12),dim=0)
            
            circuit2_input_ln = block.ln_1(circuit_2_in)# make representation matrix get normed 
            
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
            _, key_length = circuit2_input_ln.size(-2), circuit2_input_ln.size(-2)
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
            
                #get production of each head and sum of all heads
            bv_Wo=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh=torch.matmul(attn_weights,Output_mhov)+torch.matmul(attn_weights,bv_Wo)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
                circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12=Output_mh.split(1,dim=0)
            circuit_2=circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2+W_obias
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #circuit_3 is the mlp only path, 
            
            circuit_3_in=circuit_input
            if i >0:
                check_num=i*29+13
                circuit_3_in=self.check_representation(circuit_3_in,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
            
            circuit3_input_ln = block.ln_1(circuit_3_in)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            #new circuit_2 for circuit_4 
             
                
            circuit_2_in_forc4=circuit_input
            circuit_2_in_forc4=circuit_2_in_forc4.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
            circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4=\
                circuit_2_in_forc4.split(1,dim=0)
    
            if i>0:
                check_num=i*29+14
                circuit_2_in_h1_forc4=self.check_representation(circuit_2_in_h1_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+15
                circuit_2_in_h2_forc4=self.check_representation(circuit_2_in_h2_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+16
                circuit_2_in_h3_forc4=self.check_representation(circuit_2_in_h3_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+17
                circuit_2_in_h4_forc4=self.check_representation(circuit_2_in_h4_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+18
                circuit_2_in_h5_forc4=self.check_representation(circuit_2_in_h5_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+19
                circuit_2_in_h6_forc4=self.check_representation(circuit_2_in_h6_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+20
                circuit_2_in_h7_forc4=self.check_representation(circuit_2_in_h7_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+21
                circuit_2_in_h8_forc4=self.check_representation(circuit_2_in_h8_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+22
                circuit_2_in_h9_forc4=self.check_representation(circuit_2_in_h9_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+23
                circuit_2_in_h10_forc4=self.check_representation(circuit_2_in_h10_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+24
                circuit_2_in_h11_forc4=self.check_representation(circuit_2_in_h11_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+25
                circuit_2_in_h12_forc4=self.check_representation(circuit_2_in_h12_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                
            
            circuit_2_in_forc4=torch.cat(( circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4),dim=0)
            
            circuit2_input_ln_forc4 = block.ln_1(circuit_2_in_forc4)# make representation matrix get normed 
            
                #get raw attention weight A (raw compression matrix), actually A consists of 4 items
            Output_mhqk_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhqk)#X*Wqk
            Output_mhqk_forc4=torch.matmul(Output_mhqk_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#X*Wqk*XT, R^[H,N,N]=[12,14,14]
            
            Output_mhqkb1_forc4=torch.matmul(W_mhqbias,W_mhk.transpose(-1,-2))#bq*WkT
            Output_mhqkb1_forc4=torch.matmul(Output_mhqkb1_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#bq*WkT*XT, R[H,N,N]
            
            Output_mhqkb2_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhq)#X*Wq
            Output_mhqkb2_forc4=torch.matmul(Output_mhqkb2_forc4,W_mhkbias.transpose(-1,-2))#X*Wq*bkT, R[H,N,N]
            
            Output_mhqkb3_forc4=torch.matmul(W_mhqbias,W_mhkbias.transpose(-1,-2))#bq*bkT, R[H,N,N]
            
            Output_mhqk_forc4=Output_mhqk_forc4+Output_mhqkb1_forc4+Output_mhqkb2_forc4+Output_mhqkb3_forc4
            Output_mhqk_forc4 = Output_mhqk_forc4 / torch.full(
                [], 64 ** 0.5, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            
                #get compression matrix 
            # if only "normal" attention layer implements causal mask
            key_length_forc4 = circuit2_input_ln_forc4.size(-2)
            causal_mask_forc4 = torch.tril(torch.ones((key_length_forc4, key_length_forc4), dtype=torch.bool)).view(
                 1, key_length_forc4, key_length_forc4).to(self.device)
            mask_value_forc4 = torch.finfo(Output_mhqk_forc4.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value_forc4 = torch.full([], mask_value_forc4, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            Output_mhqk_forc4 = torch.where(causal_mask_forc4, Output_mhqk_forc4.to(Output_mhqk_forc4.dtype), mask_value_forc4)
            attn_weights_forc4=nn.functional.softmax(Output_mhqk_forc4, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
                #get output of OV path (representation matrix)
            Output_mhov_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            
                #get production of each head and sum of all heads
            bv_Wo_forc4=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh_forc4=torch.matmul(attn_weights_forc4,Output_mhov_forc4)+torch.matmul(attn_weights_forc4,bv_Wo_forc4)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1_forc4,circuit_2_h2_forc4,circuit_2_h3_forc4,circuit_2_h4_forc4,circuit_2_h5_forc4,circuit_2_h6_forc4,\
                circuit_2_h7_forc4,circuit_2_h8_forc4,circuit_2_h9_forc4,circuit_2_h10_forc4,circuit_2_h11_forc4,circuit_2_h12_forc4=\
                    Output_mh_forc4.split(1,dim=0)
            circuit_2_forc4=circuit_2_h1_forc4+circuit_2_h2_forc4+circuit_2_h3_forc4+circuit_2_h4_forc4+circuit_2_h5_forc4+\
                circuit_2_h6_forc4+circuit_2_h7_forc4+circuit_2_h8_forc4+circuit_2_h9_forc4+circuit_2_h10_forc4+circuit_2_h11_forc4+circuit_2_h12_forc4
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit_4_in=circuit_2_forc4+W_obias
            circuit4_input_ln = block.ln_2(circuit_4_in)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            #get subcircuit of each head and compensation circuit
            #head1
            c4h1_in=circuit_2_h1_forc4
            head1_c4in=block.ln_2(c4h1_in)
            Output_mlp1_h1=torch.matmul(head1_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h1=Act_mlp(Output_mlp1_h1) #activated
            circuit_4_h1=torch.matmul(Output_mlp1_act_4_h1,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head2
            c4h2_in=circuit_2_h2_forc4
            head2_c4in=block.ln_2(c4h2_in)
            Output_mlp1_h2=torch.matmul(head2_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h2=Act_mlp(Output_mlp1_h2) #activated
            circuit_4_h2=torch.matmul(Output_mlp1_act_4_h2,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head3 
            c4h3_in=circuit_2_h3_forc4
            head3_c4in=block.ln_2(c4h3_in)
            Output_mlp1_h3=torch.matmul(head3_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h3=Act_mlp(Output_mlp1_h3) #activated
            circuit_4_h3=torch.matmul(Output_mlp1_act_4_h3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head4
            c4h4_in=circuit_2_h4_forc4
            head4_c4in=block.ln_2(c4h4_in)
            Output_mlp1_h4=torch.matmul(head4_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h4=Act_mlp(Output_mlp1_h4) #activated
            circuit_4_h4=torch.matmul(Output_mlp1_act_4_h4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
            
            #head5
            c4h5_in=circuit_2_h5_forc4
            head5_c4in=block.ln_2(c4h5_in)
            Output_mlp1_h5=torch.matmul(head5_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h5=Act_mlp(Output_mlp1_h5) #activated
            circuit_4_h5=torch.matmul(Output_mlp1_act_4_h5,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head6
            c4h6_in=circuit_2_h6_forc4
            head6_c4in=block.ln_2(c4h6_in)
            Output_mlp1_h6=torch.matmul(head6_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h6=Act_mlp(Output_mlp1_h6) #activated
            circuit_4_h6=torch.matmul(Output_mlp1_act_4_h6,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head7
            c4h7_in=circuit_2_h7_forc4
            head7_c4in=block.ln_2(c4h7_in)
            Output_mlp1_h7=torch.matmul(head7_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h7=Act_mlp(Output_mlp1_h7) #activated
            circuit_4_h7=torch.matmul(Output_mlp1_act_4_h7,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head8
            c4h8_in=circuit_2_h8_forc4
            head8_c4in=block.ln_2(c4h8_in)
            Output_mlp1_h8=torch.matmul(head8_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h8=Act_mlp(Output_mlp1_h8) #activated
            circuit_4_h8=torch.matmul(Output_mlp1_act_4_h8,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head9
            c4h9_in=circuit_2_h9_forc4
            head9_c4in=block.ln_2(c4h9_in)
            Output_mlp1_h9=torch.matmul(head9_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h9=Act_mlp(Output_mlp1_h9) #activated
            circuit_4_h9=torch.matmul(Output_mlp1_act_4_h9,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head10
            c4h10_in=circuit_2_h10_forc4
            head10_c4in=block.ln_2(c4h10_in)
            Output_mlp1_h10=torch.matmul(head10_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h10=Act_mlp(Output_mlp1_h10) #activated
            circuit_4_h10=torch.matmul(Output_mlp1_act_4_h10,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head11
            c4h11_in=circuit_2_h11_forc4
            head11_c4in=block.ln_2(c4h11_in)
            Output_mlp1_h11=torch.matmul(head11_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h11=Act_mlp(Output_mlp1_h11) #activated
            circuit_4_h11=torch.matmul(Output_mlp1_act_4_h11,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head12
            c4h12_in=circuit_2_h12_forc4
            head12_c4in=block.ln_2(c4h12_in)
            Output_mlp1_h12=torch.matmul(head12_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h12=Act_mlp(Output_mlp1_h12) #activated
            circuit_4_h12=torch.matmul(Output_mlp1_act_4_h12,W_mlp2)#R^[B,N,d]=[1,14,768]
            
        
            
            #conpensation circuit for multi-heads, include the effects of bias in mlp1 and synergistic from interaction of multi-heads 
            
            circuit_4_compst=circuit_4-circuit_4_h1-circuit_4_h2-circuit_4_h3-circuit_4_h4-circuit_4_h5-circuit_4_h6-circuit_4_h7-circuit_4_h8-\
            circuit_4_h9-circuit_4_h10-circuit_4_h11-circuit_4_h12
            
            
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            
            circuit_5=(circuit_stream_all-circuit_3-circuit_4)
            
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wo,Wmlp1
            
            circuit_6=W_obias+W_mlp2bias
            circuit_6=circuit_6.unsqueeze(0)
            
            
            #get circuit sum 
            #circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            circuit_sum=circuit_1+circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+\
                circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12+circuit_3+circuit_4_h1+circuit_4_h2+circuit_4_h3+\
                    circuit_4_h4+circuit_4_h5+circuit_4_h6+circuit_4_h7+circuit_4_h8+circuit_4_h9+circuit_4_h10+circuit_4_h11+circuit_4_h12+\
                        circuit_4_compst+circuit_5+circuit_6
                        
            
            circuit_input=circuit_sum
            
            circuit_sum_cat=torch.cat((circuit_1,circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
            circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12,circuit_3,circuit_4_h1,circuit_4_h2,circuit_4_h3,\
                circuit_4_h4,circuit_4_h5,circuit_4_h6,circuit_4_h7,circuit_4_h8,circuit_4_h9,circuit_4_h10,circuit_4_h11,circuit_4_h12,\
                    circuit_4_compst,circuit_5,circuit_6),dim=0)#[29,N,768]
 
            record_circuit_tensor_all[i]=circuit_sum_cat
 
                    
            
        final_logits=self.get_logits(circuit_sum)
        _,predicted_indices=torch.topk(final_logits[0][-1],1)
        assert predicted_indices==label_ids
        
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def get_softmax_logits(self,input,label_ids):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        label_logits_in=F.softmax(logits_in,dim=-1).index_select(-1,label_ids)#[1,N,1]
        
        
        return label_logits_in
    
    def get_logits(self,input):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        
        return logits_in
    
    def check_representation(self,input,refine_matrix,circuit_tensor):
        to_cut_list=torch.nonzero(refine_matrix)
        for to_cut_inx in range(to_cut_list.size()[0]):
            to_cut_layer=to_cut_list[to_cut_inx].item()//29
            to_cut_circuit=to_cut_list[to_cut_inx].item()%29
            input=input-circuit_tensor[to_cut_layer][to_cut_circuit]

        return input
    
    
    



class refined_explain_model(nn.Module):
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
        self.model = self.model.to(device)
        self.layers = self.layers.to(device)


        
    
            
    def forward(self,inputs,label_ids,refined_matrix,logger):
        # refined_matrix=[349,349]
        
        
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
        
        input_ids=inputs['input_ids']
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.model.transformer.wte(input_ids)
        past_length = 0
        past_key_values = tuple([None] * len(self.layers))
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        circuit_input=hidden_states
        key_length=circuit_input.size()[-2]
        record_circuit_tensor_all=torch.zeros((12,29,key_length,768)).to(self.device)
        
    
        
        
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):

            
            #construct space mapping matrix
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
            Act_mlp=block.mlp.act #activation of mlp, and activation of attention omitted is softmax
            
            
            #circuit_1 is the self path, only include itself
            circuit_1_in=circuit_input
            if i>0:
                check_num=i*29+0
                circuit_1_in=self.check_representation(circuit_1_in,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
            circuit_1=circuit_1_in
            
            

            
            
            
            
            #circuit_2 is the attention only path, only include attention, 
            circuit_2_in=circuit_input
            circuit_2_in=circuit_2_in.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12=circuit_2_in.split(1,dim=0)
            if i>0:
                check_num=i*29+1
                circuit_2_in_h1=self.check_representation(circuit_2_in_h1,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+2
                circuit_2_in_h2=self.check_representation(circuit_2_in_h2,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+3
                circuit_2_in_h3=self.check_representation(circuit_2_in_h3,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+4
                circuit_2_in_h4=self.check_representation(circuit_2_in_h4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+5
                circuit_2_in_h5=self.check_representation(circuit_2_in_h5,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+6
                circuit_2_in_h6=self.check_representation(circuit_2_in_h6,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+7
                circuit_2_in_h7=self.check_representation(circuit_2_in_h7,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+8
                circuit_2_in_h8=self.check_representation(circuit_2_in_h8,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+9
                circuit_2_in_h9=self.check_representation(circuit_2_in_h9,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+10
                circuit_2_in_h10=self.check_representation(circuit_2_in_h10,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+11
                circuit_2_in_h11=self.check_representation(circuit_2_in_h11,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+12
                circuit_2_in_h12=self.check_representation(circuit_2_in_h12,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                
            circuit_2_in=torch.cat((circuit_2_in_h1,circuit_2_in_h2,circuit_2_in_h3,circuit_2_in_h4,circuit_2_in_h5,circuit_2_in_h6,circuit_2_in_h7,\
                circuit_2_in_h8,circuit_2_in_h9,circuit_2_in_h10,circuit_2_in_h11,circuit_2_in_h12),dim=0)
            
            circuit2_input_ln = block.ln_1(circuit_2_in)# make representation matrix get normed 
            
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
            _, key_length = circuit2_input_ln.size(-2), circuit2_input_ln.size(-2)
            causal_mask = torch.tril(torch.ones((key_length, key_length), dtype=torch.bool)).view(
                 1, key_length, key_length).to(self.device)
            mask_value = torch.finfo(Output_mhqk.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=Output_mhqk.dtype, device=Output_mhqk.device)
            Output_mhqk = torch.where(causal_mask, Output_mhqk.to(Output_mhqk.dtype), mask_value)
            attn_weights=nn.functional.softmax(Output_mhqk, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
            
            if self.args.logs=='true':
                logger.info('############showing the attention weight of each circuit')  
                self.show_attention_weight(attn_weights,logger,input_ids,i,is_mlp='false')
                
                #get output of OV path (representation matrix)
            Output_mhov=torch.matmul(circuit2_input_ln,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            
                #get production of each head and sum of all heads
            bv_Wo=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh=torch.matmul(attn_weights,Output_mhov)+torch.matmul(attn_weights,bv_Wo)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
                circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12=Output_mh.split(1,dim=0)
            circuit_2=circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2+W_obias
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #circuit_3 is the mlp only path, 
            
            circuit_3_in=circuit_input
            if i >0:
                check_num=i*29+13
                circuit_3_in=self.check_representation(circuit_3_in,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
            
            circuit3_input_ln = block.ln_1(circuit_3_in)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            #new circuit_2 for circuit_4 
             
                
            circuit_2_in_forc4=circuit_input
            circuit_2_in_forc4=circuit_2_in_forc4.repeat(12,1,1)#get multi-head representation matrix, R^[H,N,d]=[12,14,768]
            
            circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4=\
                circuit_2_in_forc4.split(1,dim=0)
    
            if i>0:
                check_num=i*29+14
                circuit_2_in_h1_forc4=self.check_representation(circuit_2_in_h1_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+15
                circuit_2_in_h2_forc4=self.check_representation(circuit_2_in_h2_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+16
                circuit_2_in_h3_forc4=self.check_representation(circuit_2_in_h3_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+17
                circuit_2_in_h4_forc4=self.check_representation(circuit_2_in_h4_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+18
                circuit_2_in_h5_forc4=self.check_representation(circuit_2_in_h5_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+19
                circuit_2_in_h6_forc4=self.check_representation(circuit_2_in_h6_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+20
                circuit_2_in_h7_forc4=self.check_representation(circuit_2_in_h7_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+21
                circuit_2_in_h8_forc4=self.check_representation(circuit_2_in_h8_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+22
                circuit_2_in_h9_forc4=self.check_representation(circuit_2_in_h9_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+23
                circuit_2_in_h10_forc4=self.check_representation(circuit_2_in_h10_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+24
                circuit_2_in_h11_forc4=self.check_representation(circuit_2_in_h11_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                check_num=i*29+25
                circuit_2_in_h12_forc4=self.check_representation(circuit_2_in_h12_forc4,refined_matrix[check_num],record_circuit_tensor_all[0:i+1])
                
            
            circuit_2_in_forc4=torch.cat(( circuit_2_in_h1_forc4,circuit_2_in_h2_forc4,circuit_2_in_h3_forc4,circuit_2_in_h4_forc4,circuit_2_in_h5_forc4,circuit_2_in_h6_forc4,\
            circuit_2_in_h7_forc4,circuit_2_in_h8_forc4,circuit_2_in_h9_forc4,circuit_2_in_h10_forc4,circuit_2_in_h11_forc4,circuit_2_in_h12_forc4),dim=0)
            
            circuit2_input_ln_forc4 = block.ln_1(circuit_2_in_forc4)# make representation matrix get normed 
            
                #get raw attention weight A (raw compression matrix), actually A consists of 4 items
            Output_mhqk_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhqk)#X*Wqk
            Output_mhqk_forc4=torch.matmul(Output_mhqk_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#X*Wqk*XT, R^[H,N,N]=[12,14,14]
            
            Output_mhqkb1_forc4=torch.matmul(W_mhqbias,W_mhk.transpose(-1,-2))#bq*WkT
            Output_mhqkb1_forc4=torch.matmul(Output_mhqkb1_forc4,circuit2_input_ln_forc4.transpose(-1,-2))#bq*WkT*XT, R[H,N,N]
            
            Output_mhqkb2_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhq)#X*Wq
            Output_mhqkb2_forc4=torch.matmul(Output_mhqkb2_forc4,W_mhkbias.transpose(-1,-2))#X*Wq*bkT, R[H,N,N]
            
            Output_mhqkb3_forc4=torch.matmul(W_mhqbias,W_mhkbias.transpose(-1,-2))#bq*bkT, R[H,N,N]
            
            Output_mhqk_forc4=Output_mhqk_forc4+Output_mhqkb1_forc4+Output_mhqkb2_forc4+Output_mhqkb3_forc4
            Output_mhqk_forc4 = Output_mhqk_forc4 / torch.full(
                [], 64 ** 0.5, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            
                #get compression matrix 
            # if only "normal" attention layer implements causal mask
            key_length_forc4 = circuit2_input_ln_forc4.size(-2)
            causal_mask_forc4 = torch.tril(torch.ones((key_length_forc4, key_length_forc4), dtype=torch.bool)).view(
                 1, key_length_forc4, key_length_forc4).to(self.device)
            mask_value_forc4 = torch.finfo(Output_mhqk_forc4.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value_forc4 = torch.full([], mask_value_forc4, dtype=Output_mhqk_forc4.dtype, device=Output_mhqk_forc4.device)
            Output_mhqk_forc4 = torch.where(causal_mask_forc4, Output_mhqk_forc4.to(Output_mhqk_forc4.dtype), mask_value_forc4)
            attn_weights_forc4=nn.functional.softmax(Output_mhqk_forc4, dim=-1) #R^[H,N,N] but R^[H,-1,N]represents the next token prediction, so the valid dim is R^[H,1,N]
            
            #show attention_weight
            if self.args.logs=='true':
                logger.info('############showing the attention weight of each circuit')  
                self.show_attention_weight(attn_weights_forc4,logger,input_ids,i,is_mlp='true')
                
                
                #get output of OV path (representation matrix)
            Output_mhov_forc4=torch.matmul(circuit2_input_ln_forc4,W_mhov)#X*Wov, R^[H,N,d]=[12,14,768]
            
                #get production of each head and sum of all heads
            bv_Wo_forc4=torch.matmul(W_mhvbias,W_mho)#R[H,N,D]=[12,14,768]
            Output_mh_forc4=torch.matmul(attn_weights_forc4,Output_mhov_forc4)+torch.matmul(attn_weights_forc4,bv_Wo_forc4)#AxWvWo+A*bv*Wo
            # R^[H,N,d], but R^[H,-1,d]represents the next token prediction, so the valid dim is R^[H,1,d]
            #get each head
            circuit_2_h1_forc4,circuit_2_h2_forc4,circuit_2_h3_forc4,circuit_2_h4_forc4,circuit_2_h5_forc4,circuit_2_h6_forc4,\
                circuit_2_h7_forc4,circuit_2_h8_forc4,circuit_2_h9_forc4,circuit_2_h10_forc4,circuit_2_h11_forc4,circuit_2_h12_forc4=\
                    Output_mh_forc4.split(1,dim=0)
            circuit_2_forc4=circuit_2_h1_forc4+circuit_2_h2_forc4+circuit_2_h3_forc4+circuit_2_h4_forc4+circuit_2_h5_forc4+\
                circuit_2_h6_forc4+circuit_2_h7_forc4+circuit_2_h8_forc4+circuit_2_h9_forc4+circuit_2_h10_forc4+circuit_2_h11_forc4+circuit_2_h12_forc4
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit_4_in=circuit_2_forc4+W_obias
            circuit4_input_ln = block.ln_2(circuit_4_in)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            #get subcircuit of each head and compensation circuit
            #head1
            c4h1_in=circuit_2_h1_forc4
            head1_c4in=block.ln_2(c4h1_in)
            Output_mlp1_h1=torch.matmul(head1_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h1=Act_mlp(Output_mlp1_h1) #activated
            circuit_4_h1=torch.matmul(Output_mlp1_act_4_h1,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head2
            c4h2_in=circuit_2_h2_forc4
            head2_c4in=block.ln_2(c4h2_in)
            Output_mlp1_h2=torch.matmul(head2_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h2=Act_mlp(Output_mlp1_h2) #activated
            circuit_4_h2=torch.matmul(Output_mlp1_act_4_h2,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head3 
            c4h3_in=circuit_2_h3_forc4
            head3_c4in=block.ln_2(c4h3_in)
            Output_mlp1_h3=torch.matmul(head3_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h3=Act_mlp(Output_mlp1_h3) #activated
            circuit_4_h3=torch.matmul(Output_mlp1_act_4_h3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head4
            c4h4_in=circuit_2_h4_forc4
            head4_c4in=block.ln_2(c4h4_in)
            Output_mlp1_h4=torch.matmul(head4_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h4=Act_mlp(Output_mlp1_h4) #activated
            circuit_4_h4=torch.matmul(Output_mlp1_act_4_h4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
            
            #head5
            c4h5_in=circuit_2_h5_forc4
            head5_c4in=block.ln_2(c4h5_in)
            Output_mlp1_h5=torch.matmul(head5_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h5=Act_mlp(Output_mlp1_h5) #activated
            circuit_4_h5=torch.matmul(Output_mlp1_act_4_h5,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head6
            c4h6_in=circuit_2_h6_forc4
            head6_c4in=block.ln_2(c4h6_in)
            Output_mlp1_h6=torch.matmul(head6_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h6=Act_mlp(Output_mlp1_h6) #activated
            circuit_4_h6=torch.matmul(Output_mlp1_act_4_h6,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head7
            c4h7_in=circuit_2_h7_forc4
            head7_c4in=block.ln_2(c4h7_in)
            Output_mlp1_h7=torch.matmul(head7_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h7=Act_mlp(Output_mlp1_h7) #activated
            circuit_4_h7=torch.matmul(Output_mlp1_act_4_h7,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head8
            c4h8_in=circuit_2_h8_forc4
            head8_c4in=block.ln_2(c4h8_in)
            Output_mlp1_h8=torch.matmul(head8_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h8=Act_mlp(Output_mlp1_h8) #activated
            circuit_4_h8=torch.matmul(Output_mlp1_act_4_h8,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head9
            c4h9_in=circuit_2_h9_forc4
            head9_c4in=block.ln_2(c4h9_in)
            Output_mlp1_h9=torch.matmul(head9_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h9=Act_mlp(Output_mlp1_h9) #activated
            circuit_4_h9=torch.matmul(Output_mlp1_act_4_h9,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head10
            c4h10_in=circuit_2_h10_forc4
            head10_c4in=block.ln_2(c4h10_in)
            Output_mlp1_h10=torch.matmul(head10_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h10=Act_mlp(Output_mlp1_h10) #activated
            circuit_4_h10=torch.matmul(Output_mlp1_act_4_h10,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head11
            c4h11_in=circuit_2_h11_forc4
            head11_c4in=block.ln_2(c4h11_in)
            Output_mlp1_h11=torch.matmul(head11_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h11=Act_mlp(Output_mlp1_h11) #activated
            circuit_4_h11=torch.matmul(Output_mlp1_act_4_h11,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head12
            c4h12_in=circuit_2_h12_forc4
            head12_c4in=block.ln_2(c4h12_in)
            Output_mlp1_h12=torch.matmul(head12_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h12=Act_mlp(Output_mlp1_h12) #activated
            circuit_4_h12=torch.matmul(Output_mlp1_act_4_h12,W_mlp2)#R^[B,N,d]=[1,14,768]
            
        
            
            #conpensation circuit for multi-heads, include the effects of bias in mlp1 and synergistic from interaction of multi-heads 
            
            circuit_4_compst=circuit_4-circuit_4_h1-circuit_4_h2-circuit_4_h3-circuit_4_h4-circuit_4_h5-circuit_4_h6-circuit_4_h7-circuit_4_h8-\
            circuit_4_h9-circuit_4_h10-circuit_4_h11-circuit_4_h12
            
            
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            
            circuit_5=(circuit_stream_all-circuit_3-circuit_4)
            
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wo,Wmlp1
            
            circuit_6=W_obias+W_mlp2bias
            circuit_6=circuit_6.unsqueeze(0)
            
            
            
            #get circuit sum 
            #circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            circuit_sum=circuit_1+circuit_2_h1+circuit_2_h2+circuit_2_h3+circuit_2_h4+circuit_2_h5+circuit_2_h6+circuit_2_h7+circuit_2_h8+\
                circuit_2_h9+circuit_2_h10+circuit_2_h11+circuit_2_h12+circuit_3+circuit_4_h1+circuit_4_h2+circuit_4_h3+\
                    circuit_4_h4+circuit_4_h5+circuit_4_h6+circuit_4_h7+circuit_4_h8+circuit_4_h9+circuit_4_h10+circuit_4_h11+circuit_4_h12+\
                        circuit_4_compst+circuit_5+circuit_6
                        
            #show each circuit's logits
            if self.args.logs=='true':
                logger.info('############showing the lable-rank of each circuit')
                self.get_circuits_rank(circuit_sum,circuit_1,circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
            circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12,circuit_3,circuit_4_h1,circuit_4_h2,circuit_4_h3,\
                circuit_4_h4,circuit_4_h5,circuit_4_h6,circuit_4_h7,circuit_4_h8,circuit_4_h9,circuit_4_h10,circuit_4_h11,circuit_4_h12,\
                    circuit_4_compst,circuit_5,circuit_6,logger,label_ids)
                        
            
            circuit_input=circuit_sum
            
            circuit_sum_cat=torch.cat((circuit_1,circuit_2_h1,circuit_2_h2,circuit_2_h3,circuit_2_h4,circuit_2_h5,circuit_2_h6,circuit_2_h7,circuit_2_h8,\
            circuit_2_h9,circuit_2_h10,circuit_2_h11,circuit_2_h12,circuit_3,circuit_4_h1,circuit_4_h2,circuit_4_h3,\
                circuit_4_h4,circuit_4_h5,circuit_4_h6,circuit_4_h7,circuit_4_h8,circuit_4_h9,circuit_4_h10,circuit_4_h11,circuit_4_h12,\
                    circuit_4_compst,circuit_5,circuit_6),dim=0)#[29,N,768]
 
            record_circuit_tensor_all[i]=circuit_sum_cat
 
                    
            
        final_logits=self.get_logits(circuit_sum)
        _,predicted_indices=torch.topk(final_logits[0][-1],1)
        assert predicted_indices==label_ids
        
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def get_softmax_logits(self,input,label_ids):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        label_logits_in=F.softmax(logits_in,dim=-1).index_select(-1,label_ids)#[1,N,1]
        
        
        return label_logits_in
    
    def get_logits(self,input):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        
        return logits_in
    
    def check_representation(self,input,refine_matrix,circuit_tensor):
        to_cut_list=torch.nonzero(refine_matrix)
        for to_cut_inx in range(to_cut_list.size()[0]):
            to_cut_layer=to_cut_list[to_cut_inx].item()//29
            to_cut_circuit=to_cut_list[to_cut_inx].item()%29
            input=input-circuit_tensor[to_cut_layer][to_cut_circuit]

        return input
    
    
    def get_logger(self,filename, verbosity=1, name=None):
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
    
    def get_token_rank(self,representation,label_ids):   
        representation=self.get_logits(representation)   
        representation=F.softmax(representation,dim=-1)
        sorted_indices=representation.argsort()
        b=sorted_indices.eq(label_ids)
        b=b.nonzero().split(1,dim=-1)[-1]
        rank =representation.size()[-1]-b
        return rank
    
    
    def get_circuits_rank(self,circuit_sum,circuit0,circuit1,circuit2,circuit3,circuit4,circuit5,circuit6,circuit7,circuit8,circuit9,circuit10,
                          circuit11,circuit12,circuit13,circuit14,circuit15,circuit16,circuit17,circuit18,circuit19,circuit20,
                          circuit21,circuit22,circuit23,circuit24,circuit25,circuit26,circuit27,circuit28,logger,label_ids):
        rank_sum=self.get_token_rank(circuit_sum,label_ids)
        logger.info('The CircuitSUM has label_rank \n {}'.format(rank_sum))
        rank_c0=self.get_token_rank(circuit0,label_ids)
        logger.info('The Circuit0 has label_rank \n {}'.format(rank_c0))
        rank_c1=self.get_token_rank(circuit1,label_ids)
        logger.info('The Circuit1 has label_rank \n {}'.format(rank_c1))
        rank_c2=self.get_token_rank(circuit2,label_ids)
        logger.info('The Circuit2 has label_rank \n {}'.format(rank_c2))
        rank_c3=self.get_token_rank(circuit3,label_ids)
        logger.info('The Circuit3 has label_rank \n {}'.format(rank_c3))
        rank_c4=self.get_token_rank(circuit4,label_ids)
        logger.info('The Circuit4 has label_rank \n {}'.format(rank_c4))
        rank_c5=self.get_token_rank(circuit5,label_ids)
        logger.info('The Circuit5 has label_rank \n {}'.format(rank_c5))
        rank_c6=self.get_token_rank(circuit6,label_ids)
        logger.info('The Circuit6 has label_rank \n {}'.format(rank_c6))
        rank_c7=self.get_token_rank(circuit7,label_ids)
        logger.info('The Circuit7 has label_rank \n {}'.format(rank_c7))
        rank_c8=self.get_token_rank(circuit8,label_ids)
        logger.info('The Circuit8 has label_rank \n {}'.format(rank_c8))
        rank_c9=self.get_token_rank(circuit9,label_ids)
        logger.info('The Circuit9 has label_rank \n {}'.format(rank_c9))
        rank_c10=self.get_token_rank(circuit10,label_ids)
        logger.info('The Circuit10 has label_rank \n {}'.format(rank_c10))
        rank_c11=self.get_token_rank(circuit11,label_ids)
        logger.info('The Circuit11 has label_rank \n {}'.format(rank_c11))
        rank_c12=self.get_token_rank(circuit12,label_ids)
        logger.info('The Circuit12 has label_rank \n {}'.format(rank_c12))
        rank_c13=self.get_token_rank(circuit13,label_ids)
        logger.info('The Circuit13 has label_rank \n {}'.format(rank_c13))
        rank_c14=self.get_token_rank(circuit14,label_ids)
        logger.info('The Circuit14 has label_rank \n {}'.format(rank_c14))
        rank_c15=self.get_token_rank(circuit15,label_ids)
        logger.info('The Circuit15 has label_rank \n {}'.format(rank_c15))
        rank_c16=self.get_token_rank(circuit16,label_ids)
        logger.info('The Circuit16 has label_rank \n {}'.format(rank_c16))
        rank_c17=self.get_token_rank(circuit17,label_ids)
        logger.info('The Circuit17 has label_rank \n {}'.format(rank_c17))
        rank_c18=self.get_token_rank(circuit18,label_ids)
        logger.info('The Circuit18 has label_rank \n {}'.format(rank_c18))
        rank_c19=self.get_token_rank(circuit19,label_ids)
        logger.info('The Circuit19 has label_rank \n {}'.format(rank_c19))
        rank_c20=self.get_token_rank(circuit20,label_ids)
        logger.info('The Circuit20 has label_rank \n {}'.format(rank_c20))
        rank_c21=self.get_token_rank(circuit21,label_ids)
        logger.info('The Circuit21 has label_rank \n {}'.format(rank_c21))
        rank_c22=self.get_token_rank(circuit22,label_ids)
        logger.info('The Circuit22 has label_rank \n {}'.format(rank_c22))
        rank_c23=self.get_token_rank(circuit23,label_ids)
        logger.info('The Circuit23 has label_rank \n {}'.format(rank_c23))
        rank_c24=self.get_token_rank(circuit24,label_ids)
        logger.info('The Circuit24 has label_rank \n {}'.format(rank_c24))
        rank_c25=self.get_token_rank(circuit25,label_ids)
        logger.info('The Circuit25 has label_rank \n {}'.format(rank_c25))
        rank_c26=self.get_token_rank(circuit26,label_ids)
        logger.info('The Circuit26 has label_rank \n {}'.format(rank_c26))
        rank_c27=self.get_token_rank(circuit27,label_ids)
        logger.info('The Circuit27 has label_rank \n {}'.format(rank_c27))
        rank_c28=self.get_token_rank(circuit28,label_ids)
        logger.info('The Circuit28 has label_rank \n {}'.format(rank_c28))
        
        
    def show_attention_weight(self,attention_weight,logger,input_ids,i,is_mlp):
        head1_weight, head2_weight,head3_weight, head4_weight,head5_weight, head6_weight,head7_weight, head8_weight,\
                head9_weight, head10_weight,head11_weight, head12_weight=attention_weight.split(1,dim=0)#[1,N,N]
        if is_mlp=='false':
            for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head1 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head1_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            # for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head2 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head2_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head3 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head3_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head4 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head4_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head5 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head5_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head6 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head6_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head7 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head7_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head8 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head8_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head9 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head9_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head10 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head10_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head11 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head11_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head12 weight for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head12_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
        
        if is_mlp=='true':
            for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head1 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head1_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            # for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head2 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head2_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head3 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head3_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head4 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head4_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
            
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head5 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head5_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head6 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head6_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head7 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head7_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head8 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head8_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head9 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head9_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head10 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head10_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head11 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head11_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )
                
            #for token in range(input_ids.size(-1)):
                logger.info('##{}-th layer ##Weight##: The head12 weight before mlp for token ['.format(i)+self.tokenizer.decode(input_ids[0][token])+'] are: {} for source tokens ['.format(head12_weight[0][token][:token+1].data)+self.tokenizer.decode(input_ids[0][:token+1])+']' )