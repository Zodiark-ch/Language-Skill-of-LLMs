import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer,GPT2LMHeadModel
from tqdm import trange
import numpy as np
import logging



class attention_circuit(nn.Module):
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
        
        logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+'_logging.log')
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            logger.info('Start to show the embedding space of attention in {}-th layer'.format(i))
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
            circuit_2=head1_attn+head2_attn+head3_attn+head4_attn+head5_attn+head6_attn+head7_attn+head8_attn+head9_attn+head10_attn+head11_attn+head12_attn
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            #get each circuit embedding matrix 
            
            #the whole output
            ln_hidden_state_all=self.model.transformer.ln_f(hidden_states)
            circuit_all_logits=self.model.lm_head(ln_hidden_state_all)[0][-1].unsqueeze(0)
            
            #the circuit 1
            ln_hidden_state_c1=self.model.transformer.ln_f(circuit_1)
            circuit_1_logits=self.model.lm_head(ln_hidden_state_c1)[0][-1].unsqueeze(0)
            
            #the circuit 2
            ln_hidden_state_c2=self.model.transformer.ln_f(circuit_2)
            circuit_2_logits=self.model.lm_head(ln_hidden_state_c2)[0][-1].unsqueeze(0)
            
            #each head
            ln_hidden_state_h1=self.model.transformer.ln_f(head1_attn)
            head1_logits=self.model.lm_head(ln_hidden_state_h1)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h2=self.model.transformer.ln_f(head2_attn)
            head2_logits=self.model.lm_head(ln_hidden_state_h2)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h3=self.model.transformer.ln_f(head3_attn)
            head3_logits=self.model.lm_head(ln_hidden_state_h3)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h4=self.model.transformer.ln_f(head4_attn)
            head4_logits=self.model.lm_head(ln_hidden_state_h4)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h5=self.model.transformer.ln_f(head5_attn)
            head5_logits=self.model.lm_head(ln_hidden_state_h5)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h6=self.model.transformer.ln_f(head6_attn)
            head6_logits=self.model.lm_head(ln_hidden_state_h6)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h7=self.model.transformer.ln_f(head7_attn)
            head7_logits=self.model.lm_head(ln_hidden_state_h7)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h8=self.model.transformer.ln_f(head8_attn)
            head8_logits=self.model.lm_head(ln_hidden_state_h8)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h9=self.model.transformer.ln_f(head9_attn)
            head9_logits=self.model.lm_head(ln_hidden_state_h9)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h10=self.model.transformer.ln_f(head10_attn)
            head10_logits=self.model.lm_head(ln_hidden_state_h10)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h11=self.model.transformer.ln_f(head11_attn)
            head11_logits=self.model.lm_head(ln_hidden_state_h11)[0][-1].unsqueeze(0)
            
            ln_hidden_state_h12=self.model.transformer.ln_f(head12_attn)
            head12_logits=self.model.lm_head(ln_hidden_state_h12)[0][-1].unsqueeze(0)
            
            
            #get the KLD from the whole output
            distribution_all=F.softmax(circuit_all_logits,dim=-1)
            distribution_c1=F.log_softmax(circuit_1_logits)
            distribution_c2=F.log_softmax(circuit_2_logits)
            distribution_h1=F.log_softmax(head1_logits)
            distribution_h2=F.log_softmax(head2_logits)
            distribution_h3=F.log_softmax(head3_logits)
            distribution_h4=F.log_softmax(head4_logits)
            distribution_h5=F.log_softmax(head5_logits)
            distribution_h6=F.log_softmax(head6_logits)
            distribution_h7=F.log_softmax(head7_logits)
            distribution_h8=F.log_softmax(head8_logits)
            distribution_h9=F.log_softmax(head9_logits)
            distribution_h10=F.log_softmax(head10_logits)
            distribution_h11=F.log_softmax(head11_logits)
            distribution_h12=F.log_softmax(head12_logits)
            
            kld_c1=F.kl_div(distribution_c1,distribution_all,reduction='sum')
            kld_c2=F.kl_div(distribution_c2,distribution_all,reduction='sum')
            kld_h1=F.kl_div(distribution_h1,distribution_all,reduction='sum')
            kld_h2=F.kl_div(distribution_h2,distribution_all,reduction='sum')
            kld_h3=F.kl_div(distribution_h3,distribution_all,reduction='sum')
            kld_h4=F.kl_div(distribution_h4,distribution_all,reduction='sum')
            kld_h5=F.kl_div(distribution_h5,distribution_all,reduction='sum')
            kld_h6=F.kl_div(distribution_h6,distribution_all,reduction='sum')
            kld_h7=F.kl_div(distribution_h7,distribution_all,reduction='sum')
            kld_h8=F.kl_div(distribution_h8,distribution_all,reduction='sum')
            kld_h9=F.kl_div(distribution_h9,distribution_all,reduction='sum')
            kld_h10=F.kl_div(distribution_h10,distribution_all,reduction='sum')
            kld_h11=F.kl_div(distribution_h11,distribution_all,reduction='sum')
            kld_h12=F.kl_div(distribution_h12,distribution_all,reduction='sum')
            head_list=[kld_h1,kld_h2,kld_h3,kld_h4,kld_h5,kld_h6,kld_h7,kld_h8,kld_h9,kld_h10,kld_h11,kld_h12]
            kld_min_head=head_list.index(min(head_list))
            
            logger.info('##{}-th layer ##KLD##: The KL(C1||all) is {}'.format(i,kld_c1))
            logger.info('##{}-th layer ##KLD##: The KL(C2||all) is {}'.format(i,kld_c2))
            
            logger.info('##{}-th layer ##KLD##: The KL(H1||all) is {}'.format(i,kld_h1))
            logger.info('##{}-th layer ##KLD##: The KL(H2||all) is {}'.format(i,kld_h2))
            logger.info('##{}-th layer ##KLD##: The KL(H3||all) is {}'.format(i,kld_h3))
            logger.info('##{}-th layer ##KLD##: The KL(H4||all) is {}'.format(i,kld_h4))
            logger.info('##{}-th layer ##KLD##: The KL(H5||all) is {}'.format(i,kld_h5))
            logger.info('##{}-th layer ##KLD##: The KL(H6||all) is {}'.format(i,kld_h6))
            logger.info('##{}-th layer ##KLD##: The KL(H7||all) is {}'.format(i,kld_h7))
            logger.info('##{}-th layer ##KLD##: The KL(H8||all) is {}'.format(i,kld_h8))
            logger.info('##{}-th layer ##KLD##: The KL(H9||all) is {}'.format(i,kld_h9))
            logger.info('##{}-th layer ##KLD##: The KL(H10||all) is {}'.format(i,kld_h10))
            logger.info('##{}-th layer ##KLD##: The KL(H11||all) is {}'.format(i,kld_h11))
            logger.info('##{}-th layer ##KLD##: The KL(H12||all) is {}'.format(i,kld_h12))
            logger.info('##{}-th layer ##KLD##: The most minimal KLD head is head-{} with value {}'.format(i,kld_min_head+1,head_list[kld_min_head]))
            
            
            #get the top 10 token
            _,predicted_indices_all=torch.topk(circuit_all_logits,10)
            _,predicted_indices_c1=torch.topk(circuit_1_logits,10)
            _,predicted_indices_c2=torch.topk(circuit_2_logits,10)
            _,predicted_indices_h1=torch.topk(head1_logits,10)
            _,predicted_indices_h2=torch.topk(head2_logits,10)
            _,predicted_indices_h3=torch.topk(head3_logits,10)
            _,predicted_indices_h4=torch.topk(head4_logits,10)
            _,predicted_indices_h5=torch.topk(head5_logits,10)
            _,predicted_indices_h6=torch.topk(head6_logits,10)
            _,predicted_indices_h7=torch.topk(head7_logits,10)
            _,predicted_indices_h8=torch.topk(head8_logits,10)
            _,predicted_indices_h9=torch.topk(head9_logits,10)
            _,predicted_indices_h10=torch.topk(head10_logits,10)
            _,predicted_indices_h11=torch.topk(head11_logits,10)
            _,predicted_indices_h12=torch.topk(head12_logits,10)
            
            logger.info('##{}-th layer ##Token##: The top 10 tokens of all are {}'.format(i,self.tokenizer.decode(predicted_indices_all[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of c1 are {}'.format(i,self.tokenizer.decode(predicted_indices_c1[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of c2 are {}'.format(i,self.tokenizer.decode(predicted_indices_c2[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h1 are {}'.format(i,self.tokenizer.decode(predicted_indices_h1[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h2 are {}'.format(i,self.tokenizer.decode(predicted_indices_h2[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h3 are {}'.format(i,self.tokenizer.decode(predicted_indices_h3[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h4 are {}'.format(i,self.tokenizer.decode(predicted_indices_h4[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h5 are {}'.format(i,self.tokenizer.decode(predicted_indices_h5[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h6 are {}'.format(i,self.tokenizer.decode(predicted_indices_h6[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h7 are {}'.format(i,self.tokenizer.decode(predicted_indices_h7[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h8 are {}'.format(i,self.tokenizer.decode(predicted_indices_h8[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h9 are {}'.format(i,self.tokenizer.decode(predicted_indices_h9[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h10 are {}'.format(i,self.tokenizer.decode(predicted_indices_h10[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h11 are {}'.format(i,self.tokenizer.decode(predicted_indices_h11[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of h12 are {}'.format(i,self.tokenizer.decode(predicted_indices_h12[0])))

            
            
            #get the top 10 tokens of representation through OV circuit
            head1_ov,head2_ov,head3_ov,head4_ov,head5_ov,head6_ov,head7_ov,head8_ov,\
                head9_ov,head10_ov,head11_ov,head12_ov=Output_mhov.split(1,dim=0)
                
            ln_ov_h1=self.model.transformer.ln_f(head1_ov)
            ov_head1_logits=self.model.lm_head(ln_ov_h1)[0][-1].unsqueeze(0)
            ln_ov_h2=self.model.transformer.ln_f(head2_ov)
            ov_head2_logits=self.model.lm_head(ln_ov_h2)[0][-1].unsqueeze(0)
            ln_ov_h3=self.model.transformer.ln_f(head3_ov)
            ov_head3_logits=self.model.lm_head(ln_ov_h3)[0][-1].unsqueeze(0)
            ln_ov_h4=self.model.transformer.ln_f(head4_ov)
            ov_head4_logits=self.model.lm_head(ln_ov_h4)[0][-1].unsqueeze(0)
            ln_ov_h5=self.model.transformer.ln_f(head5_ov)
            ov_head5_logits=self.model.lm_head(ln_ov_h5)[0][-1].unsqueeze(0)
            ln_ov_h6=self.model.transformer.ln_f(head6_ov)
            ov_head6_logits=self.model.lm_head(ln_ov_h6)[0][-1].unsqueeze(0)
            ln_ov_h7=self.model.transformer.ln_f(head7_ov)
            ov_head7_logits=self.model.lm_head(ln_ov_h7)[0][-1].unsqueeze(0)
            ln_ov_h8=self.model.transformer.ln_f(head8_ov)
            ov_head8_logits=self.model.lm_head(ln_ov_h8)[0][-1].unsqueeze(0)
            ln_ov_h9=self.model.transformer.ln_f(head9_ov)
            ov_head9_logits=self.model.lm_head(ln_ov_h9)[0][-1].unsqueeze(0)
            ln_ov_h10=self.model.transformer.ln_f(head10_ov)
            ov_head10_logits=self.model.lm_head(ln_ov_h10)[0][-1].unsqueeze(0)
            ln_ov_h11=self.model.transformer.ln_f(head11_ov)
            ov_head11_logits=self.model.lm_head(ln_ov_h11)[0][-1].unsqueeze(0)
            ln_ov_h12=self.model.transformer.ln_f(head12_ov)
            ov_head12_logits=self.model.lm_head(ln_ov_h12)[0][-1].unsqueeze(0)
            
            
            _,predicted_indices_ov_h1=torch.topk(ov_head1_logits,10)
            _,predicted_indices_ov_h2=torch.topk(ov_head2_logits,10)
            _,predicted_indices_ov_h3=torch.topk(ov_head3_logits,10)
            _,predicted_indices_ov_h4=torch.topk(ov_head4_logits,10)
            _,predicted_indices_ov_h5=torch.topk(ov_head5_logits,10)
            _,predicted_indices_ov_h6=torch.topk(ov_head6_logits,10)
            _,predicted_indices_ov_h7=torch.topk(ov_head7_logits,10)
            _,predicted_indices_ov_h8=torch.topk(ov_head8_logits,10)
            _,predicted_indices_ov_h9=torch.topk(ov_head9_logits,10)
            _,predicted_indices_ov_h10=torch.topk(ov_head10_logits,10)
            _,predicted_indices_ov_h11=torch.topk(ov_head11_logits,10)
            _,predicted_indices_ov_h12=torch.topk(ov_head12_logits,10)
            
        
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h1 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h1[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h2 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h2[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h3 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h3[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h4 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h4[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h5 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h5[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h6 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h6[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h7 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h7[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h8 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h8[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h9 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h9[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h10 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h10[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h11 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h11[0])))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of OV path in h12 are {}'.format(i,self.tokenizer.decode(predicted_indices_ov_h12[0])))
            
            
            
            #show the attention weights 
            token_list=self.tokenizer.decode(input_ids[0])
            head1_weight, head2_weight,head3_weight, head4_weight,head5_weight, head6_weight,head7_weight, head8_weight,\
                head9_weight, head10_weight,head11_weight, head12_weight=attn_weights.split(1,dim=0)#[1,N,N]
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
            
            
            
            #show circuit_sum vocabualry
            ln_hidden_state=self.model.transformer.ln_f(hidden_states)
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
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
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
                       
                       
                       
                       
class ioi_attention_circuit(nn.Module):
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


        
    
            
    def forward(self,inputs,input_text,word_idx,IO,IOm1,IOa1,S,Sm1,Sa1,S2):
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
        
        
        duplicate_weight=torch.zeros((len(self.layers),12))
        induction_weight=torch.zeros((len(self.layers),12))
        previous_weight=torch.zeros((len(self.layers),12))
        Name_weight=torch.zeros((len(self.layers),12))
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
            circuit_2=head1_attn+head2_attn+head3_attn+head4_attn+head5_attn+head6_attn+head7_attn+head8_attn+head9_attn+head10_attn+head11_attn+head12_attn
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            #get each circuit embedding matrix 
            
            #the whole output
            ln_hidden_state_all=self.model.transformer.ln_f(hidden_states)
            circuit_all_logits=self.model.lm_head(ln_hidden_state_all)[0][-1].unsqueeze(0)
            
            #the circuit 1
            ln_hidden_state_c1=self.model.transformer.ln_f(circuit_1)
            circuit_1_logits=self.model.lm_head(ln_hidden_state_c1)[0][-1].unsqueeze(0)
            
            #the circuit 2
            ln_hidden_state_c2=self.model.transformer.ln_f(circuit_2)
            circuit_2_logits=self.model.lm_head(ln_hidden_state_c2)[0][-1].unsqueeze(0)
            
            
            
            #show the attention weights 
            token_list=self.tokenizer.decode(input_ids[0])
            head1_weight, head2_weight,head3_weight, head4_weight,head5_weight, head6_weight,head7_weight, head8_weight,\
                head9_weight, head10_weight,head11_weight, head12_weight=attn_weights.split(1,dim=0)#[1,N,N]
            
            
            
            # find duplicate token heads 
            duplicate_weight[i][0]=head1_weight[0][S2][S]
            duplicate_weight[i][1]=head2_weight[0][S2][S]
            duplicate_weight[i][2]=head3_weight[0][S2][S]
            duplicate_weight[i][3]=head4_weight[0][S2][S]
            duplicate_weight[i][4]=head5_weight[0][S2][S]
            duplicate_weight[i][5]=head6_weight[0][S2][S]
            duplicate_weight[i][6]=head7_weight[0][S2][S]
            duplicate_weight[i][7]=head8_weight[0][S2][S]
            duplicate_weight[i][8]=head9_weight[0][S2][S]
            duplicate_weight[i][9]=head10_weight[0][S2][S]
            duplicate_weight[i][10]=head11_weight[0][S2][S]
            duplicate_weight[i][11]=head12_weight[0][S2][S]
            
            #find induction heads
            induction_weight[i][0]=head1_weight[0][S2][Sa1]
            induction_weight[i][1]=head2_weight[0][S2][Sa1]
            induction_weight[i][2]=head3_weight[0][S2][Sa1]
            induction_weight[i][3]=head4_weight[0][S2][Sa1]
            induction_weight[i][4]=head5_weight[0][S2][Sa1]
            induction_weight[i][5]=head6_weight[0][S2][Sa1]
            induction_weight[i][6]=head7_weight[0][S2][Sa1]
            induction_weight[i][7]=head8_weight[0][S2][Sa1]
            induction_weight[i][8]=head9_weight[0][S2][Sa1]
            induction_weight[i][9]=head10_weight[0][S2][Sa1]
            induction_weight[i][10]=head11_weight[0][S2][Sa1]
            induction_weight[i][11]=head12_weight[0][S2][Sa1]
            
            #find previous heads
            previous_weight[i][0]=head1_weight[0][-1][S2]
            previous_weight[i][1]=head2_weight[0][-1][S2]
            previous_weight[i][2]=head3_weight[0][-1][S2]
            previous_weight[i][3]=head4_weight[0][-1][S2]
            previous_weight[i][4]=head5_weight[0][-1][S2]
            previous_weight[i][5]=head6_weight[0][-1][S2]
            previous_weight[i][6]=head7_weight[0][-1][S2]
            previous_weight[i][7]=head8_weight[0][-1][S2]
            previous_weight[i][8]=head9_weight[0][-1][S2]
            previous_weight[i][9]=head10_weight[0][-1][S2]
            previous_weight[i][10]=head11_weight[0][-1][S2]
            previous_weight[i][11]=head12_weight[0][-1][S2]
            
            #find name heads
            Name_weight[i][0]=head1_weight[0][-1][S2]
            Name_weight[i][1]=head2_weight[0][-1][S2]
            Name_weight[i][2]=head3_weight[0][-1][S2]
            Name_weight[i][3]=head4_weight[0][-1][S2]
            Name_weight[i][4]=head5_weight[0][-1][S2]
            Name_weight[i][5]=head6_weight[0][-1][S2]
            Name_weight[i][6]=head7_weight[0][-1][S2]
            Name_weight[i][7]=head8_weight[0][-1][S2]
            Name_weight[i][8]=head9_weight[0][-1][S2]
            Name_weight[i][9]=head10_weight[0][-1][S2]
            Name_weight[i][10]=head11_weight[0][-1][S2]
            Name_weight[i][11]=head12_weight[0][-1][S2]
            
            
            
        return duplicate_weight,induction_weight,previous_weight,Name_weight
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
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
                       
                       
class circuit_analysis(nn.Module):
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
        logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+self.tokenizer.decode(input_ids[0])+'_logging.log')
        logger.info('Input text is:'+self.tokenizer.decode(input_ids[0])) 
        cos_matrix=torch.zeros((12,6))
        mse_matrix=torch.zeros((12,6))
        jsd_matrix=torch.zeros((12,6))
        ce_matrix=torch.zeros((12,6))
        top_cos_matrix=torch.zeros((12,6))
        top_mse_matrix=torch.zeros((12,6))
        top_jsd_matrix=torch.zeros((12,6))
        top_ce_matrix=torch.zeros((12,6))
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
            circuit_2=head1_attn+head2_attn+head3_attn+head4_attn+head5_attn+head6_attn+head7_attn+head8_attn+head9_attn+head10_attn+head11_attn+head12_attn
            #finally add the bias of Wo, because Wo is conducted after merging the head
            
            
            #get the activation mapping 
            residual_stream=circuit_1+circuit_2
            circuit3_input_ln = block.ln_2(residual_stream)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1_all=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_all_act_steam=Act_mlp(Output_mlp1_all) #activated
            circuit_stream_all=torch.matmul(Output_mlp1_all_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            # Output_mlp1_act_steam=Act_mlp(Output_mlp1_all-W_mlp1bias) #activated
            # circuit_stream=torch.matmul(Output_mlp1_act_steam,W_mlp2)#R^[B,N,d]=[1,14,768]
            # circuit_Wmlp1bias=circuit_stream_all-circuit_stream
            # Output_mlp1_bias=Act_mlp(W_mlp1bias) #activated
            # circuit_uni_wmlp1bias=torch.matmul(Output_mlp1_bias,W_mlp2)#R^[B,N,d]=[1,14,768]
            # circuit_syn_bias=circuit_Wmlp1bias-circuit_uni_wmlp1bias
            
            
            
            
            #circuit_3 is the mlp only path, 
            circuit3_input_ln = block.ln_1(circuit_input)# make representation matrix get normed R^[N,d]=[14,768]
            circuit3_input_ln = block.ln_2(circuit3_input_ln)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit4_input_ln = block.ln_2(circuit_2)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1)+W_mlp1bias #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
    
            
            # circuit_5, the effect of addition of circuit_1 and circuit_2 caused by NewGeLU activation, also, 
            # meaning that the synergistic of residual stream (syn(A,B), and syn((A+B),Wmlp1bias))
            circuit_5=(circuit_stream_all-circuit_3-circuit_4)
            
            #circuit_6, i.e.,circuit_Wmlp1bias, the movement of bias in Wo,Wmlp1
            circuit_6=W_obias+W_mlp2bias
            
            #get circuit sum 
            circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            
            #show circuit_sum vocabualry
            ln_hidden_state=self.model.transformer.ln_f(circuit_sum)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            logger.info('In {} layer output, max probability tokens of circuit_sum are:'.format(i)+self.tokenizer.decode(predicted_indices[0])) 
            
            
            #get the cossim of each circuit 
            tokens_c1,tokens_c2,tokens_c2h1,tokens_c2h2,tokens_c2h3,tokens_c2h4,tokens_c2h5,tokens_c2h6,tokens_c2h7,tokens_c2h8,\
                tokens_c2h9,tokens_c2h10,tokens_c2h11,tokens_c2h12,tokens_c3,tokens_c4,tokens_c5,tokens_c6,\
cos_sim_c1,cos_sim_c2,cos_sim_c2h1,cos_sim_c2h2,cos_sim_c2h3,cos_sim_c2h4,cos_sim_c2h5,cos_sim_c2h6,cos_sim_c2h7,cos_sim_c2h8,cos_sim_c2h9,cos_sim_c2h10,\
cos_sim_c2h11,cos_sim_c2h12,cos_sim_c3,cos_sim_c4,cos_sim_c5,cos_sim_c6,\
    top_cos_sim_c1,top_cos_sim_c2,top_cos_sim_c2h1,top_cos_sim_c2h2,top_cos_sim_c2h3,top_cos_sim_c2h4,top_cos_sim_c2h5,top_cos_sim_c2h6,\
    top_cos_sim_c2h7,top_cos_sim_c2h8,top_cos_sim_c2h9,top_cos_sim_c2h10,top_cos_sim_c2h11,top_cos_sim_c2h12,top_cos_sim_c3,top_cos_sim_c4,\
        top_cos_sim_c5,top_cos_sim_c6,\
mse_c1,mse_c2,mse_c3,mse_c4,mse_c5,mse_c6,mse_c2h1,mse_c2h2,mse_c2h3,mse_c2h4,mse_c2h5,mse_c2h6,mse_c2h7,mse_c2h8,mse_c2h9,mse_c2h10,mse_c2h11,mse_c2h12,\
top_mse_c1,top_mse_c2,top_mse_c3,top_mse_c4,top_mse_c5,top_mse_c6,top_mse_c2h1,top_mse_c2h2,top_mse_c2h3,top_mse_c2h4,top_mse_c2h5,top_mse_c2h6,\
    top_mse_c2h7,top_mse_c2h8,top_mse_c2h9,top_mse_c2h10,top_mse_c2h11,top_mse_c2h12,\
        ce_c1,ce_c2,ce_c3,ce_c4,ce_c5,ce_c6,ce_c2h1,ce_c2h2,ce_c2h3,ce_c2h4,ce_c2h5,ce_c2h6,ce_c2h7,ce_c2h8,ce_c2h9,ce_c2h10,ce_c2h11,ce_c2h12,\
top_ce_c1,top_ce_c2,top_ce_c3,top_ce_c4,top_ce_c5,top_ce_c6,top_ce_c2h1,top_ce_c2h2,top_ce_c2h3,top_ce_c2h4,top_ce_c2h5,top_ce_c2h6,\
    top_ce_c2h7,top_ce_c2h8,top_ce_c2h9,top_ce_c2h10,top_ce_c2h11,top_ce_c2h12,\
kld_c1,kld_c2,kld_c3,kld_c4,kld_c5,kld_c6,kld_h1,kld_h2,kld_h3,kld_h4,kld_h5,kld_h6,kld_h7,kld_h8,kld_h9,kld_h10,kld_h11,kld_h12,\
top_kld_c1,top_kld_c2,top_kld_c3,top_kld_c4,top_kld_c5,top_kld_c6,top_kld_h1,top_kld_h2,top_kld_h3,top_kld_h4,top_kld_h5,top_kld_h6,top_kld_h7,\
    top_kld_h8,top_kld_h9,top_kld_h10,top_kld_h11,top_kld_h12, \
        kld_min_head,top_kld_min_head=\
                             self.get_token_distance(circuit_logits,\
                            circuit_1,circuit_2,circuit_3,circuit_4,circuit_5,circuit_6,head1_attn,head2_attn,head3_attn,\
                                head4_attn,head5_attn,head6_attn,head7_attn,head8_attn,head9_attn,head10_attn,head11_attn,head12_attn)
            
            
            
            
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C1 path are {}'.format(i,tokens_c1))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2 path are {}'.format(i,tokens_c2))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C3 path are {}'.format(i,tokens_c3))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C4 path are {}'.format(i,tokens_c4))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C5 path are {}'.format(i,tokens_c5))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C6 path are {}'.format(i,tokens_c6))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h1 path are {}'.format(i,tokens_c2h1))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h2 path are {}'.format(i,tokens_c2h2))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h3 path are {}'.format(i,tokens_c2h3))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h4 path are {}'.format(i,tokens_c2h4))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h5 path are {}'.format(i,tokens_c2h5))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h6 path are {}'.format(i,tokens_c2h6))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h7 path are {}'.format(i,tokens_c2h7))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h8 path are {}'.format(i,tokens_c2h8))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h9 path are {}'.format(i,tokens_c2h9))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h10 path are {}'.format(i,tokens_c2h10))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h11 path are {}'.format(i,tokens_c2h11))
            logger.info('##{}-th layer ##Token##: The top 10 tokens of C2h12 path are {}'.format(i,tokens_c2h12))
            
            logger.info('##{}-th layer ##Token##: The COSSIM of C1 path are {}'.format(i,cos_sim_c1))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2 path are {}'.format(i,cos_sim_c2))
            logger.info('##{}-th layer ##Token##: The COSSIM of C3 path are {}'.format(i,cos_sim_c3))
            logger.info('##{}-th layer ##Token##: The COSSIM of C4 path are {}'.format(i,cos_sim_c4))
            logger.info('##{}-th layer ##Token##: The COSSIM of C5 path are {}'.format(i,cos_sim_c5))
            logger.info('##{}-th layer ##Token##: The COSSIM of C6 path are {}'.format(i,cos_sim_c6))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H1 path are {}'.format(i,cos_sim_c2h1))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H2 path are {}'.format(i,cos_sim_c2h2))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H3 path are {}'.format(i,cos_sim_c2h3))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H4 path are {}'.format(i,cos_sim_c2h4))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H5 path are {}'.format(i,cos_sim_c2h5))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H6 path are {}'.format(i,cos_sim_c2h6))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H7 path are {}'.format(i,cos_sim_c2h7))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H8 path are {}'.format(i,cos_sim_c2h8))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H9 path are {}'.format(i,cos_sim_c2h9))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H10 path are {}'.format(i,cos_sim_c2h10))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H11 path are {}'.format(i,cos_sim_c2h11))
            logger.info('##{}-th layer ##Token##: The COSSIM of C2H12 path are {}'.format(i,cos_sim_c2h12))
            cos_sim_list=[cos_sim_c1,cos_sim_c2,cos_sim_c3,cos_sim_c4,cos_sim_c5,cos_sim_c6]
            cos_sim_max=cos_sim_list.index(max(cos_sim_list))
            cos_matrix[i][cos_sim_max]=1
            
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C1 path are {}'.format(i,top_cos_sim_c1))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2 path are {}'.format(i,top_cos_sim_c2))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C3 path are {}'.format(i,top_cos_sim_c3))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C4 path are {}'.format(i,top_cos_sim_c4))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C5 path are {}'.format(i,top_cos_sim_c5))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C6 path are {}'.format(i,top_cos_sim_c6))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H1 path are {}'.format(i,top_cos_sim_c2h1))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H2 path are {}'.format(i,top_cos_sim_c2h2))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H3 path are {}'.format(i,top_cos_sim_c2h3))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H4 path are {}'.format(i,top_cos_sim_c2h4))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H5 path are {}'.format(i,top_cos_sim_c2h5))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H6 path are {}'.format(i,top_cos_sim_c2h6))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H7 path are {}'.format(i,top_cos_sim_c2h7))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H8 path are {}'.format(i,top_cos_sim_c2h8))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H9 path are {}'.format(i,top_cos_sim_c2h9))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H10 path are {}'.format(i,top_cos_sim_c2h10))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H11 path are {}'.format(i,top_cos_sim_c2h11))
            logger.info('##{}-th layer ##Token##: The COSSIM of TOP50-C2H12 path are {}'.format(i,top_cos_sim_c2h12))
            top_cos_sim_list=[top_cos_sim_c1,top_cos_sim_c2,top_cos_sim_c3,top_cos_sim_c4,top_cos_sim_c5,top_cos_sim_c6]
            top_cos_sim_max=top_cos_sim_list.index(max(top_cos_sim_list))
            top_cos_matrix[i][top_cos_sim_max]=1
            
            logger.info('##{}-th layer ##Token##: The MSE of C1 path are {}'.format(i,mse_c1))
            logger.info('##{}-th layer ##Token##: The MSE of C2 path are {}'.format(i,mse_c2))
            logger.info('##{}-th layer ##Token##: The MSE of C3 path are {}'.format(i,mse_c3))
            logger.info('##{}-th layer ##Token##: The MSE of C4 path are {}'.format(i,mse_c4))
            logger.info('##{}-th layer ##Token##: The MSE of C5 path are {}'.format(i,mse_c5))
            logger.info('##{}-th layer ##Token##: The MSE of C6 path are {}'.format(i,mse_c6))
            logger.info('##{}-th layer ##Token##: The MSE of C2H1 path are {}'.format(i,mse_c2h1))
            logger.info('##{}-th layer ##Token##: The MSE of C2H2 path are {}'.format(i,mse_c2h2))
            logger.info('##{}-th layer ##Token##: The MSE of C2H3 path are {}'.format(i,mse_c2h3))
            logger.info('##{}-th layer ##Token##: The MSE of C2H4 path are {}'.format(i,mse_c2h4))
            logger.info('##{}-th layer ##Token##: The MSE of C2H5 path are {}'.format(i,mse_c2h5))
            logger.info('##{}-th layer ##Token##: The MSE of C2H6 path are {}'.format(i,mse_c2h6))
            logger.info('##{}-th layer ##Token##: The MSE of C2H7 path are {}'.format(i,mse_c2h7))
            logger.info('##{}-th layer ##Token##: The MSE of C2H8 path are {}'.format(i,mse_c2h8))
            logger.info('##{}-th layer ##Token##: The MSE of C2H9 path are {}'.format(i,mse_c2h9))
            logger.info('##{}-th layer ##Token##: The MSE of C2H10 path are {}'.format(i,mse_c2h10))
            logger.info('##{}-th layer ##Token##: The MSE of C2H11 path are {}'.format(i,mse_c2h11))
            logger.info('##{}-th layer ##Token##: The MSE of C2H12 path are {}'.format(i,mse_c2h12))
            mse_list=[mse_c1,mse_c2,mse_c3,mse_c4,mse_c5,mse_c6]
            mse_max=mse_list.index(min(mse_list))
            mse_matrix[i][mse_max]=1
            
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C1 path are {}'.format(i,top_mse_c1))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2 path are {}'.format(i,top_mse_c2))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C3 path are {}'.format(i,top_mse_c3))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C4 path are {}'.format(i,top_mse_c4))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C5 path are {}'.format(i,top_mse_c5))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C6 path are {}'.format(i,top_mse_c6))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H1 path are {}'.format(i,top_mse_c2h1))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H2 path are {}'.format(i,top_mse_c2h2))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H3 path are {}'.format(i,top_mse_c2h3))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H4 path are {}'.format(i,top_mse_c2h4))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H5 path are {}'.format(i,top_mse_c2h5))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H6 path are {}'.format(i,top_mse_c2h6))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H7 path are {}'.format(i,top_mse_c2h7))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H8 path are {}'.format(i,top_mse_c2h8))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H9 path are {}'.format(i,top_mse_c2h9))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H10 path are {}'.format(i,top_mse_c2h10))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H11 path are {}'.format(i,top_mse_c2h11))
            logger.info('##{}-th layer ##Token##: The MSE of TOP50-C2H12 path are {}'.format(i,top_mse_c2h12))
            top_mse_list=[top_mse_c1,top_mse_c2,top_mse_c3,top_mse_c4,top_mse_c5,top_mse_c6]
            top_mse_max=top_mse_list.index(min(top_mse_list))
            top_mse_matrix[i][top_mse_max]=1
            
            logger.info('##{}-th layer ##Token##: The CE of C1 path are {}'.format(i,ce_c1))
            logger.info('##{}-th layer ##Token##: The CE of C2 path are {}'.format(i,ce_c2))
            logger.info('##{}-th layer ##Token##: The CE of C3 path are {}'.format(i,ce_c3))
            logger.info('##{}-th layer ##Token##: The CE of C4 path are {}'.format(i,ce_c4))
            logger.info('##{}-th layer ##Token##: The CE of C5 path are {}'.format(i,ce_c5))
            logger.info('##{}-th layer ##Token##: The CE of C6 path are {}'.format(i,ce_c6))
            logger.info('##{}-th layer ##Token##: The CE of C2H1 path are {}'.format(i,ce_c2h1))
            logger.info('##{}-th layer ##Token##: The CE of C2H2 path are {}'.format(i,ce_c2h2))
            logger.info('##{}-th layer ##Token##: The CE of C2H3 path are {}'.format(i,ce_c2h3))
            logger.info('##{}-th layer ##Token##: The CE of C2H4 path are {}'.format(i,ce_c2h4))
            logger.info('##{}-th layer ##Token##: The CE of C2H5 path are {}'.format(i,ce_c2h5))
            logger.info('##{}-th layer ##Token##: The CE of C2H6 path are {}'.format(i,ce_c2h6))
            logger.info('##{}-th layer ##Token##: The CE of C2H7 path are {}'.format(i,ce_c2h7))
            logger.info('##{}-th layer ##Token##: The CE of C2H8 path are {}'.format(i,ce_c2h8))
            logger.info('##{}-th layer ##Token##: The CE of C2H9 path are {}'.format(i,ce_c2h9))
            logger.info('##{}-th layer ##Token##: The CE of C2H10 path are {}'.format(i,ce_c2h10))
            logger.info('##{}-th layer ##Token##: The CE of C2H11 path are {}'.format(i,ce_c2h11))
            logger.info('##{}-th layer ##Token##: The CE of C2H12 path are {}'.format(i,ce_c2h12))
            ce_list=[ce_c1,ce_c2,ce_c3,ce_c4,ce_c5,ce_c6]
            ce_max=ce_list.index(min(ce_list))
            ce_matrix[i][ce_max]=1
            
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C1 path are {}'.format(i,top_ce_c1))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2 path are {}'.format(i,top_ce_c2))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C3 path are {}'.format(i,top_ce_c3))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C4 path are {}'.format(i,top_ce_c4))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C5 path are {}'.format(i,top_ce_c5))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C6 path are {}'.format(i,top_ce_c6))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H1 path are {}'.format(i,top_ce_c2h1))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H2 path are {}'.format(i,top_ce_c2h2))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H3 path are {}'.format(i,top_ce_c2h3))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H4 path are {}'.format(i,top_ce_c2h4))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H5 path are {}'.format(i,top_ce_c2h5))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H6 path are {}'.format(i,top_ce_c2h6))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H7 path are {}'.format(i,top_ce_c2h7))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H8 path are {}'.format(i,top_ce_c2h8))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H9 path are {}'.format(i,top_ce_c2h9))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H10 path are {}'.format(i,top_ce_c2h10))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H11 path are {}'.format(i,top_ce_c2h11))
            logger.info('##{}-th layer ##Token##: The CE of TOP50-C2H12 path are {}'.format(i,top_ce_c2h12))
            top_ce_list=[top_ce_c1,top_ce_c2,top_ce_c3,top_ce_c4,top_ce_c5,top_ce_c6]
            top_ce_max=top_ce_list.index(min(top_ce_list))
            top_ce_matrix[i][top_ce_max]=1
            
            logger.info('##{}-th layer ##JSD##: The JSD(C1||all) is {}'.format(i,kld_c1))
            logger.info('##{}-th layer ##JSD##: The JSD(C2||all) is {}'.format(i,kld_c2))
            logger.info('##{}-th layer ##JSD##: The JSD(C3||all) is {}'.format(i,kld_c3))
            logger.info('##{}-th layer ##JSD##: The JSD(C4||all) is {}'.format(i,kld_c4))
            logger.info('##{}-th layer ##JSD##: The JSD(C5||all) is {}'.format(i,kld_c5))
            logger.info('##{}-th layer ##JSD##: The JSD(C6||all) is {}'.format(i,kld_c6))
            
            logger.info('##{}-th layer ##JSD##: The JSD(H1||all) is {}'.format(i,kld_h1))
            logger.info('##{}-th layer ##JSD##: The JSD(H2||all) is {}'.format(i,kld_h2))
            logger.info('##{}-th layer ##JSD##: The JSD(H3||all) is {}'.format(i,kld_h3))
            logger.info('##{}-th layer ##JSD##: The JSD(H4||all) is {}'.format(i,kld_h4))
            logger.info('##{}-th layer ##JSD##: The JSD(H5||all) is {}'.format(i,kld_h5))
            logger.info('##{}-th layer ##JSD##: The JSD(H6||all) is {}'.format(i,kld_h6))
            logger.info('##{}-th layer ##JSD##: The JSD(H7||all) is {}'.format(i,kld_h7))
            logger.info('##{}-th layer ##JSD##: The JSD(H8||all) is {}'.format(i,kld_h8))
            logger.info('##{}-th layer ##JSD##: The JSD(H9||all) is {}'.format(i,kld_h9))
            logger.info('##{}-th layer ##JSD##: The JSD(H10||all) is {}'.format(i,kld_h10))
            logger.info('##{}-th layer ##JSD##: The JSD(H11||all) is {}'.format(i,kld_h11))
            logger.info('##{}-th layer ##JSD##: The JSD(H12||all) is {}'.format(i,kld_h12))
            logger.info('##{}-th layer ##JSD##: The most minimal JSD head is head{}'.format(i,kld_min_head+1))
            jsd_list=[kld_c1,kld_c2,kld_c3,kld_c4,kld_c5,kld_c6]
            jsd_max=jsd_list.index(min(jsd_list))
            jsd_matrix[i][jsd_max]=1
            
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C1||all) is {}'.format(i,top_kld_c1))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C2||all) is {}'.format(i,top_kld_c2))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C3||all) is {}'.format(i,top_kld_c3))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C4||all) is {}'.format(i,top_kld_c4))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C5||all) is {}'.format(i,top_kld_c5))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(C6||all) is {}'.format(i,top_kld_c6))
            
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H1||all) is {}'.format(i,top_kld_h1))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H2||all) is {}'.format(i,top_kld_h2))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H3||all) is {}'.format(i,top_kld_h3))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H4||all) is {}'.format(i,top_kld_h4))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H5||all) is {}'.format(i,top_kld_h5))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H6||all) is {}'.format(i,top_kld_h6))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H7||all) is {}'.format(i,top_kld_h7))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H8||all) is {}'.format(i,top_kld_h8))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H9||all) is {}'.format(i,top_kld_h9))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H10||all) is {}'.format(i,top_kld_h10))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H11||all) is {}'.format(i,top_kld_h11))
            logger.info('##{}-th layer ##JSD##: The TOP50 JSD(H12||all) is {}'.format(i,top_kld_h12))
            logger.info('##{}-th layer ##JSD##: The most minimal TOP50 JSD head is head{}'.format(i,top_kld_min_head+1))
            top_jsd_list=[top_kld_c1,top_kld_c2,top_kld_c3,top_kld_c4,top_kld_c5,top_kld_c6]
            top_jsd_max=top_jsd_list.index(min(top_jsd_list))
            top_jsd_matrix[i][top_jsd_max]=1
            
        logger.info('The cos_matrix_all matrix is {}'.format(cos_matrix))
        logger.info('The top_cos_matrix_all matrix is {}'.format(top_cos_matrix))
        logger.info('The mse_matrix_all matrix is {}'.format(mse_matrix))
        logger.info('The top_mse_matrix_all matrix is {}'.format(top_mse_matrix))
        logger.info('The ce_matrix_all matrix is {}'.format(ce_matrix))
        logger.info('The top_ce_matrix_all matrix is {}'.format(top_ce_matrix))
        logger.info('The jsd_matrix_all matrix is {}'.format(jsd_matrix))
        logger.info('The top_jsd_matrix_all matrix is {}'.format(top_jsd_matrix))
            
        logging.shutdown()
        return cos_matrix,top_cos_matrix,mse_matrix,top_mse_matrix,ce_matrix,top_ce_matrix,jsd_matrix,top_jsd_matrix
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    
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
    
    def get_tokens(self,predicted_indices):
        token_list=[]
        for i in range(10):
            ids=predicted_indices[0][i]
            token_list.append(self.tokenizer.decode(ids))
        return token_list
    
    def get_token_distance(self,circuit_logits,circuit_1,circuit_2,circuit_3,circuit_4,circuit_5,circuit_6,\
        head1_attn,head2_attn,head3_attn,head4_attn,head5_attn,head6_attn,head7_attn,head8_attn,head9_attn,head10_attn,head11_attn,head12_attn):
            #get top_50 of circuit_logits
            top_circuit_logit,top_circuit_idx=torch.topk(circuit_logits,50)
            
            
            
            
            
            #circuit_1
                #show circuit_1 vocabualry and cossim
            ln_hidden_state_c1=self.model.transformer.ln_f(circuit_1)
            circuit_logits_c1=self.model.lm_head(ln_hidden_state_c1)[0][-1].unsqueeze(0)
            _,predicted_indices_c1=torch.topk(circuit_logits_c1,10)
            tokens_c1=self.get_tokens(predicted_indices_c1)
            cos_sim_c1=F.cosine_similarity(F.softmax(circuit_logits_c1,dim=-1),F.softmax(circuit_logits))
            mse_c1=F.mse_loss(F.softmax(circuit_logits_c1,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c1=circuit_logits_c1.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c1=F.cosine_similarity(F.softmax(top_circuit_logit_c1,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c1=F.mse_loss(F.softmax(top_circuit_logit_c1,dim=-1),F.softmax(top_circuit_logit))
            ce_c1=F.cross_entropy(F.softmax(circuit_logits_c1,dim=-1),F.softmax(circuit_logits))
            top_ce_c1=F.cross_entropy(F.softmax(top_circuit_logit_c1,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_2
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2=self.model.transformer.ln_f(circuit_2)
            circuit_logits_c2=self.model.lm_head(ln_hidden_state_c2)[0][-1].unsqueeze(0)
            _,predicted_indices_c2=torch.topk(circuit_logits_c2,10)
            tokens_c2=self.get_tokens(predicted_indices_c2)
            cos_sim_c2=F.cosine_similarity(F.softmax(circuit_logits_c2,dim=-1),F.softmax(circuit_logits))
            mse_c2=F.mse_loss(F.softmax(circuit_logits_c2,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2=circuit_logits_c2.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2=F.cosine_similarity(F.softmax(top_circuit_logit_c2,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2=F.mse_loss(F.softmax(top_circuit_logit_c2,dim=-1),F.softmax(top_circuit_logit))
            ce_c2=F.cross_entropy(F.softmax(circuit_logits_c2,dim=-1),F.softmax(circuit_logits))
            top_ce_c2=F.cross_entropy(F.softmax(top_circuit_logit_c2,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h1
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h1=self.model.transformer.ln_f(head1_attn)
            circuit_logits_c2h1=self.model.lm_head(ln_hidden_state_c2h1)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h1=torch.topk(circuit_logits_c2h1,10)
            tokens_c2h1=self.get_tokens(predicted_indices_c2h1)
            cos_sim_c2h1=F.cosine_similarity(F.softmax(circuit_logits_c2h1,dim=-1),F.softmax(circuit_logits))
            mse_c2h1=F.mse_loss(F.softmax(circuit_logits_c2h1,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h1=circuit_logits_c2h1.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h1=F.cosine_similarity(F.softmax(top_circuit_logit_c2h1,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h1=F.mse_loss(F.softmax(top_circuit_logit_c2h1,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h1=F.cross_entropy(F.softmax(circuit_logits_c2h1,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h1=F.cross_entropy(F.softmax(top_circuit_logit_c2h1,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h2
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h2=self.model.transformer.ln_f(head2_attn)
            circuit_logits_c2h2=self.model.lm_head(ln_hidden_state_c2h2)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h2=torch.topk(circuit_logits_c2h2,10)
            tokens_c2h2=self.get_tokens(predicted_indices_c2h2)
            cos_sim_c2h2=F.cosine_similarity(F.softmax(circuit_logits_c2h2,dim=-1),F.softmax(circuit_logits))
            mse_c2h2=F.mse_loss(F.softmax(circuit_logits_c2h2,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h2=circuit_logits_c2h2.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h2=F.cosine_similarity(F.softmax(top_circuit_logit_c2h2,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h2=F.mse_loss(F.softmax(top_circuit_logit_c2h2,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h2=F.cross_entropy(F.softmax(circuit_logits_c2h2,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h2=F.cross_entropy(F.softmax(top_circuit_logit_c2h2,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h3
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h3=self.model.transformer.ln_f(head3_attn)
            circuit_logits_c2h3=self.model.lm_head(ln_hidden_state_c2h3)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h3=torch.topk(circuit_logits_c2h3,10)
            tokens_c2h3=self.get_tokens(predicted_indices_c2h3)
            cos_sim_c2h3=F.cosine_similarity(F.softmax(circuit_logits_c2h3,dim=-1),F.softmax(circuit_logits))
            mse_c2h3=F.mse_loss(F.softmax(circuit_logits_c2h3,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h3=circuit_logits_c2h3.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h3=F.cosine_similarity(F.softmax(top_circuit_logit_c2h3,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h3=F.mse_loss(F.softmax(top_circuit_logit_c2h3,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h3=F.cross_entropy(F.softmax(circuit_logits_c2h3,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h3=F.cross_entropy(F.softmax(top_circuit_logit_c2h3,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h4
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h4=self.model.transformer.ln_f(head4_attn)
            circuit_logits_c2h4=self.model.lm_head(ln_hidden_state_c2h4)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h4=torch.topk(circuit_logits_c2h4,10)
            tokens_c2h4=self.get_tokens(predicted_indices_c2h4)
            cos_sim_c2h4=F.cosine_similarity(F.softmax(circuit_logits_c2h4,dim=-1),F.softmax(circuit_logits))
            mse_c2h4=F.mse_loss(F.softmax(circuit_logits_c2h4,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h4=circuit_logits_c2h4.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h4=F.cosine_similarity(F.softmax(top_circuit_logit_c2h4,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h4=F.mse_loss(F.softmax(top_circuit_logit_c2h4,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h4=F.cross_entropy(F.softmax(circuit_logits_c2h4,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h4=F.cross_entropy(F.softmax(top_circuit_logit_c2h4,dim=-1),F.softmax(top_circuit_logit))
            
            
            
                #circuit_2h5
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h5=self.model.transformer.ln_f(head5_attn)
            circuit_logits_c2h5=self.model.lm_head(ln_hidden_state_c2h5)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h5=torch.topk(circuit_logits_c2h5,10)
            tokens_c2h5=self.get_tokens(predicted_indices_c2h5)
            cos_sim_c2h5=F.cosine_similarity(F.softmax(circuit_logits_c2h5,dim=-1),F.softmax(circuit_logits))
            mse_c2h5=F.mse_loss(F.softmax(circuit_logits_c2h5,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h5=circuit_logits_c2h5.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h5=F.cosine_similarity(F.softmax(top_circuit_logit_c2h5,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h5=F.mse_loss(F.softmax(top_circuit_logit_c2h5,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h5=F.cross_entropy(F.softmax(circuit_logits_c2h5,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h5=F.cross_entropy(F.softmax(top_circuit_logit_c2h5,dim=-1),F.softmax(top_circuit_logit))
            
            
            
                #circuit_2h6
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h6=self.model.transformer.ln_f(head6_attn)
            circuit_logits_c2h6=self.model.lm_head(ln_hidden_state_c2h6)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h6=torch.topk(circuit_logits_c2h6,10)
            tokens_c2h6=self.get_tokens(predicted_indices_c2h6)
            cos_sim_c2h6=F.cosine_similarity(F.softmax(circuit_logits_c2h6,dim=-1),F.softmax(circuit_logits))
            mse_c2h6=F.mse_loss(F.softmax(circuit_logits_c2h6,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h6=circuit_logits_c2h6.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h6=F.cosine_similarity(F.softmax(top_circuit_logit_c2h6,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h6=F.mse_loss(F.softmax(top_circuit_logit_c2h6,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h6=F.cross_entropy(F.softmax(circuit_logits_c2h6,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h6=F.cross_entropy(F.softmax(top_circuit_logit_c2h6,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h7
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h7=self.model.transformer.ln_f(head7_attn)
            circuit_logits_c2h7=self.model.lm_head(ln_hidden_state_c2h7)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h7=torch.topk(circuit_logits_c2h7,10)
            tokens_c2h7=self.get_tokens(predicted_indices_c2h7)
            cos_sim_c2h7=F.cosine_similarity(F.softmax(circuit_logits_c2h7,dim=-1),F.softmax(circuit_logits))
            mse_c2h7=F.mse_loss(F.softmax(circuit_logits_c2h7,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h7=circuit_logits_c2h7.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h7=F.cosine_similarity(F.softmax(top_circuit_logit_c2h7,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h7=F.mse_loss(F.softmax(top_circuit_logit_c2h7,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h7=F.cross_entropy(F.softmax(circuit_logits_c2h7,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h7=F.cross_entropy(F.softmax(top_circuit_logit_c2h7,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h8
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h8=self.model.transformer.ln_f(head8_attn)
            circuit_logits_c2h8=self.model.lm_head(ln_hidden_state_c2h8)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h8=torch.topk(circuit_logits_c2h8,10)
            tokens_c2h8=self.get_tokens(predicted_indices_c2h8)
            cos_sim_c2h8=F.cosine_similarity(F.softmax(circuit_logits_c2h8,dim=-1),F.softmax(circuit_logits))
            mse_c2h8=F.mse_loss(F.softmax(circuit_logits_c2h8,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h8=circuit_logits_c2h8.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h8=F.cosine_similarity(F.softmax(top_circuit_logit_c2h8,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h8=F.mse_loss(F.softmax(top_circuit_logit_c2h8,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h8=F.cross_entropy(F.softmax(circuit_logits_c2h8,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h8=F.cross_entropy(F.softmax(top_circuit_logit_c2h8,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h9
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h9=self.model.transformer.ln_f(head9_attn)
            circuit_logits_c2h9=self.model.lm_head(ln_hidden_state_c2h9)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h9=torch.topk(circuit_logits_c2h9,10)
            tokens_c2h9=self.get_tokens(predicted_indices_c2h9)
            cos_sim_c2h9=F.cosine_similarity(F.softmax(circuit_logits_c2h9,dim=-1),F.softmax(circuit_logits))
            mse_c2h9=F.mse_loss(F.softmax(circuit_logits_c2h9,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h9=circuit_logits_c2h9.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h9=F.cosine_similarity(F.softmax(top_circuit_logit_c2h9,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h9=F.mse_loss(F.softmax(top_circuit_logit_c2h9,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h9=F.cross_entropy(F.softmax(circuit_logits_c2h9,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h9=F.cross_entropy(F.softmax(top_circuit_logit_c2h9,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h10
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h10=self.model.transformer.ln_f(head10_attn)
            circuit_logits_c2h10=self.model.lm_head(ln_hidden_state_c2h10)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h10=torch.topk(circuit_logits_c2h10,10)
            tokens_c2h10=self.get_tokens(predicted_indices_c2h10)
            cos_sim_c2h10=F.cosine_similarity(F.softmax(circuit_logits_c2h10,dim=-1),F.softmax(circuit_logits))
            mse_c2h10=F.mse_loss(F.softmax(circuit_logits_c2h10,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h10=circuit_logits_c2h10.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h10=F.cosine_similarity(F.softmax(top_circuit_logit_c2h10,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h10=F.mse_loss(F.softmax(top_circuit_logit_c2h10,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h10=F.cross_entropy(F.softmax(circuit_logits_c2h10,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h10=F.cross_entropy(F.softmax(top_circuit_logit_c2h10,dim=-1),F.softmax(top_circuit_logit))
            
            
                #circuit_2h11
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h11=self.model.transformer.ln_f(head11_attn)
            circuit_logits_c2h11=self.model.lm_head(ln_hidden_state_c2h11)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h11=torch.topk(circuit_logits_c2h11,10)
            tokens_c2h11=self.get_tokens(predicted_indices_c2h11)
            cos_sim_c2h11=F.cosine_similarity(F.softmax(circuit_logits_c2h11,dim=-1),F.softmax(circuit_logits))
            mse_c2h11=F.mse_loss(F.softmax(circuit_logits_c2h11,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h11=circuit_logits_c2h11.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h11=F.cosine_similarity(F.softmax(top_circuit_logit_c2h11,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h11=F.mse_loss(F.softmax(top_circuit_logit_c2h11,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h11=F.cross_entropy(F.softmax(circuit_logits_c2h11,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h11=F.cross_entropy(F.softmax(top_circuit_logit_c2h11,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_2h12
                #show circuit_2 vocabualry and cossim
            ln_hidden_state_c2h12=self.model.transformer.ln_f(head12_attn)
            circuit_logits_c2h12=self.model.lm_head(ln_hidden_state_c2h12)[0][-1].unsqueeze(0)
            _,predicted_indices_c2h12=torch.topk(circuit_logits_c2h12,10)
            tokens_c2h12=self.get_tokens(predicted_indices_c2h12)
            cos_sim_c2h12=F.cosine_similarity(F.softmax(circuit_logits_c2h12,dim=-1),F.softmax(circuit_logits))
            mse_c2h12=F.mse_loss(F.softmax(circuit_logits_c2h12,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c2h12=circuit_logits_c2h12.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c2h12=F.cosine_similarity(F.softmax(top_circuit_logit_c2h12,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c2h12=F.mse_loss(F.softmax(top_circuit_logit_c2h12,dim=-1),F.softmax(top_circuit_logit))
            ce_c2h12=F.cross_entropy(F.softmax(circuit_logits_c2h12,dim=-1),F.softmax(circuit_logits))
            top_ce_c2h12=F.cross_entropy(F.softmax(top_circuit_logit_c2h12,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_3
                #show circuit_3 vocabualry and cossim
            ln_hidden_state_c3=self.model.transformer.ln_f(circuit_3)
            circuit_logits_c3=self.model.lm_head(ln_hidden_state_c3)[0][-1].unsqueeze(0)
            _,predicted_indices_c3=torch.topk(circuit_logits_c3,10)
            tokens_c3=self.get_tokens(predicted_indices_c3)
            cos_sim_c3=F.cosine_similarity(F.softmax(circuit_logits_c3,dim=-1),F.softmax(circuit_logits))
            mse_c3=F.mse_loss(F.softmax(circuit_logits_c3,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c3=circuit_logits_c3.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c3=F.cosine_similarity(F.softmax(top_circuit_logit_c3,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c3=F.mse_loss(F.softmax(top_circuit_logit_c3,dim=-1),F.softmax(top_circuit_logit))
            ce_c3=F.cross_entropy(F.softmax(circuit_logits_c3,dim=-1),F.softmax(circuit_logits))
            top_ce_c3=F.cross_entropy(F.softmax(top_circuit_logit_c3,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_4
                #show circuit_4 vocabualry and cossim
            ln_hidden_state_c4=self.model.transformer.ln_f(circuit_4)
            circuit_logits_c4=self.model.lm_head(ln_hidden_state_c4)[0][-1].unsqueeze(0)
            _,predicted_indices_c4=torch.topk(circuit_logits_c4,10)
            tokens_c4=self.get_tokens(predicted_indices_c4)
            cos_sim_c4=F.cosine_similarity(F.softmax(circuit_logits_c4,dim=-1),F.softmax(circuit_logits))
            mse_c4=F.mse_loss(F.softmax(circuit_logits_c4,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c4=circuit_logits_c4.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c4=F.cosine_similarity(F.softmax(top_circuit_logit_c4,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c4=F.mse_loss(F.softmax(top_circuit_logit_c4,dim=-1),F.softmax(top_circuit_logit))
            ce_c4=F.cross_entropy(F.softmax(circuit_logits_c4,dim=-1),F.softmax(circuit_logits))
            top_ce_c4=F.cross_entropy(F.softmax(top_circuit_logit_c4,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_5
                #show circuit_5 vocabualry and cossim
            ln_hidden_state_c5=self.model.transformer.ln_f(circuit_5)
            circuit_logits_c5=self.model.lm_head(ln_hidden_state_c5)[0][-1].unsqueeze(0)
            _,predicted_indices_c5=torch.topk(circuit_logits_c5,10)
            tokens_c5=self.get_tokens(predicted_indices_c5)
            cos_sim_c5=F.cosine_similarity(F.softmax(circuit_logits_c5,dim=-1),F.softmax(circuit_logits))
            mse_c5=F.mse_loss(F.softmax(circuit_logits_c5,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c5=circuit_logits_c5.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c5=F.cosine_similarity(F.softmax(top_circuit_logit_c5,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c5=F.mse_loss(F.softmax(top_circuit_logit_c5,dim=-1),F.softmax(top_circuit_logit))
            ce_c5=F.cross_entropy(F.softmax(circuit_logits_c5,dim=-1),F.softmax(circuit_logits))
            top_ce_c5=F.cross_entropy(F.softmax(top_circuit_logit_c5,dim=-1),F.softmax(top_circuit_logit))
            
                #circuit_6
                #show circuit_6 vocabualry and cossim
            ln_hidden_state_c6=self.model.transformer.ln_f(circuit_6)
            circuit_logits_c6=self.model.lm_head(ln_hidden_state_c6)[-1].unsqueeze(0)
            _,predicted_indices_c6=torch.topk(circuit_logits_c6,10)
            tokens_c6=self.get_tokens(predicted_indices_c6)
            cos_sim_c6=F.cosine_similarity(F.softmax(circuit_logits_c6,dim=-1),F.softmax(circuit_logits))
            mse_c6=F.mse_loss(F.softmax(circuit_logits_c6,dim=-1),F.softmax(circuit_logits))
            top_circuit_logit_c6=circuit_logits_c6.index_select(-1,top_circuit_idx[0])
            top_cos_sim_c6=F.cosine_similarity(F.softmax(top_circuit_logit_c6,dim=-1),F.softmax(top_circuit_logit))
            top_mse_c6=F.mse_loss(F.softmax(top_circuit_logit_c6,dim=-1),F.softmax(top_circuit_logit))
            ce_c6=F.cross_entropy(F.softmax(circuit_logits_c6,dim=-1),F.softmax(circuit_logits))
            top_ce_c6=F.cross_entropy(F.softmax(top_circuit_logit_c6,dim=-1),F.softmax(top_circuit_logit))
            
            
            #get the JSD from the whole output
            
            distribution_all_c1=F.softmax((circuit_logits+circuit_logits_c1)/circuit_logits,dim=-1)
            distribution_all_c2=F.softmax((circuit_logits+circuit_logits_c2)/circuit_logits,dim=-1)
            distribution_all_c3=F.softmax((circuit_logits+circuit_logits_c3)/circuit_logits,dim=-1)
            distribution_all_c4=F.softmax((circuit_logits+circuit_logits_c4)/circuit_logits,dim=-1)
            distribution_all_c5=F.softmax((circuit_logits+circuit_logits_c5)/circuit_logits,dim=-1)
            distribution_all_c6=F.softmax((circuit_logits+circuit_logits_c6)/circuit_logits,dim=-1)
            
            distribution_all_c2h1=F.softmax((circuit_logits+circuit_logits_c2h1)/circuit_logits,dim=-1)
            distribution_all_c2h2=F.softmax((circuit_logits+circuit_logits_c2h2)/circuit_logits,dim=-1)
            distribution_all_c2h3=F.softmax((circuit_logits+circuit_logits_c2h3)/circuit_logits,dim=-1)
            distribution_all_c2h4=F.softmax((circuit_logits+circuit_logits_c2h4)/circuit_logits,dim=-1)
            distribution_all_c2h5=F.softmax((circuit_logits+circuit_logits_c2h5)/circuit_logits,dim=-1)
            distribution_all_c2h6=F.softmax((circuit_logits+circuit_logits_c2h6)/circuit_logits,dim=-1)
            distribution_all_c2h7=F.softmax((circuit_logits+circuit_logits_c2h7)/circuit_logits,dim=-1)
            distribution_all_c2h8=F.softmax((circuit_logits+circuit_logits_c2h8)/circuit_logits,dim=-1)
            distribution_all_c2h9=F.softmax((circuit_logits+circuit_logits_c2h9)/circuit_logits,dim=-1)
            distribution_all_c2h10=F.softmax((circuit_logits+circuit_logits_c2h10)/circuit_logits,dim=-1)
            distribution_all_c2h11=F.softmax((circuit_logits+circuit_logits_c2h11)/circuit_logits,dim=-1)
            distribution_all_c2h12=F.softmax((circuit_logits+circuit_logits_c2h12)/circuit_logits,dim=-1)
            
            distribution_all=F.log_softmax(circuit_logits,dim=-1)
            distribution_c1=F.log_softmax(circuit_logits_c1)
            distribution_c2=F.log_softmax(circuit_logits_c2)
            distribution_c3=F.log_softmax(circuit_logits_c3)
            distribution_c4=F.log_softmax(circuit_logits_c4)
            distribution_c5=F.log_softmax(circuit_logits_c5)
            distribution_c6=F.log_softmax(circuit_logits_c6)
            distribution_h1=F.log_softmax(circuit_logits_c2h1)
            distribution_h2=F.log_softmax(circuit_logits_c2h2)
            distribution_h3=F.log_softmax(circuit_logits_c2h3)
            distribution_h4=F.log_softmax(circuit_logits_c2h4)
            distribution_h5=F.log_softmax(circuit_logits_c2h5)
            distribution_h6=F.log_softmax(circuit_logits_c2h6)
            distribution_h7=F.log_softmax(circuit_logits_c2h7)
            distribution_h8=F.log_softmax(circuit_logits_c2h8)
            distribution_h9=F.log_softmax(circuit_logits_c2h9)
            distribution_h10=F.log_softmax(circuit_logits_c2h10)
            distribution_h11=F.log_softmax(circuit_logits_c2h11)
            distribution_h12=F.log_softmax(circuit_logits_c2h12)
            
            
            kld_c1=0.5*F.kl_div(distribution_c1,distribution_all_c1,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c1,reduction='sum')
            kld_c2=0.5*F.kl_div(distribution_c2,distribution_all_c2,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2,reduction='sum')
            kld_c3=0.5*F.kl_div(distribution_c3,distribution_all_c3,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c3,reduction='sum')
            kld_c4=0.5*F.kl_div(distribution_c4,distribution_all_c4,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c4,reduction='sum')
            kld_c5=0.5*F.kl_div(distribution_c5,distribution_all_c5,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c5,reduction='sum')
            kld_c6=0.5*F.kl_div(distribution_c6,distribution_all_c6,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c6,reduction='sum')
            kld_h1=0.5*F.kl_div(distribution_h1,distribution_all_c2h1,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h1,reduction='sum')
            kld_h2=0.5*F.kl_div(distribution_h2,distribution_all_c2h2,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h2,reduction='sum')
            kld_h3=0.5*F.kl_div(distribution_h3,distribution_all_c2h3,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h3,reduction='sum')
            kld_h4=0.5*F.kl_div(distribution_h4,distribution_all_c2h4,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h4,reduction='sum')
            kld_h5=0.5*F.kl_div(distribution_h5,distribution_all_c2h5,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h5,reduction='sum')
            kld_h6=0.5*F.kl_div(distribution_h6,distribution_all_c2h6,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h6,reduction='sum')
            kld_h7=0.5*F.kl_div(distribution_h7,distribution_all_c2h7,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h7,reduction='sum')
            kld_h8=0.5*F.kl_div(distribution_h8,distribution_all_c2h8,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h8,reduction='sum')
            kld_h9=0.5*F.kl_div(distribution_h9,distribution_all_c2h9,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h9,reduction='sum')
            kld_h10=0.5*F.kl_div(distribution_h10,distribution_all_c2h10,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h10,reduction='sum')
            kld_h11=0.5*F.kl_div(distribution_h11,distribution_all_c2h11,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h11,reduction='sum')
            kld_h12=0.5*F.kl_div(distribution_h12,distribution_all_c2h12,reduction='sum')+0.5*F.kl_div(distribution_all,distribution_all_c2h12,reduction='sum')
            head_list=[kld_h1,kld_h2,kld_h3,kld_h4,kld_h5,kld_h6,kld_h7,kld_h8,kld_h9,kld_h10,kld_h11,kld_h12]
            kld_min_head=head_list.index(min(head_list))
            
            
            #get the top-JSD from the whole output
            
            top_distribution_all_c1=F.softmax((top_circuit_logit+top_circuit_logit_c1)/top_circuit_logit,dim=-1)
            top_distribution_all_c2=F.softmax((top_circuit_logit+top_circuit_logit_c2)/top_circuit_logit,dim=-1)
            top_distribution_all_c3=F.softmax((top_circuit_logit+top_circuit_logit_c3)/top_circuit_logit,dim=-1)
            top_distribution_all_c4=F.softmax((top_circuit_logit+top_circuit_logit_c4)/top_circuit_logit,dim=-1)
            top_distribution_all_c5=F.softmax((top_circuit_logit+top_circuit_logit_c5)/top_circuit_logit,dim=-1)
            top_distribution_all_c6=F.softmax((top_circuit_logit+top_circuit_logit_c6)/top_circuit_logit,dim=-1)
            
            top_distribution_all_c2h1=F.softmax((top_circuit_logit+top_circuit_logit_c2h1)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h2=F.softmax((top_circuit_logit+top_circuit_logit_c2h2)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h3=F.softmax((top_circuit_logit+top_circuit_logit_c2h3)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h4=F.softmax((top_circuit_logit+top_circuit_logit_c2h4)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h5=F.softmax((top_circuit_logit+top_circuit_logit_c2h5)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h6=F.softmax((top_circuit_logit+top_circuit_logit_c2h6)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h7=F.softmax((top_circuit_logit+top_circuit_logit_c2h7)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h8=F.softmax((top_circuit_logit+top_circuit_logit_c2h8)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h9=F.softmax((top_circuit_logit+top_circuit_logit_c2h9)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h10=F.softmax((top_circuit_logit+top_circuit_logit_c2h10)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h11=F.softmax((top_circuit_logit+top_circuit_logit_c2h11)/top_circuit_logit,dim=-1)
            top_distribution_all_c2h12=F.softmax((top_circuit_logit+top_circuit_logit_c2h12)/top_circuit_logit,dim=-1)
            
            top_distribution_all=F.log_softmax(top_circuit_logit,dim=-1)
            top_distribution_c1=F.log_softmax(top_circuit_logit_c1)
            top_distribution_c2=F.log_softmax(top_circuit_logit_c2)
            top_distribution_c3=F.log_softmax(top_circuit_logit_c3)
            top_distribution_c4=F.log_softmax(top_circuit_logit_c4)
            top_distribution_c5=F.log_softmax(top_circuit_logit_c5)
            top_distribution_c6=F.log_softmax(top_circuit_logit_c6)
            top_distribution_h1=F.log_softmax(top_circuit_logit_c2h1)
            top_distribution_h2=F.log_softmax(top_circuit_logit_c2h2)
            top_distribution_h3=F.log_softmax(top_circuit_logit_c2h3)
            top_distribution_h4=F.log_softmax(top_circuit_logit_c2h4)
            top_distribution_h5=F.log_softmax(top_circuit_logit_c2h5)
            top_distribution_h6=F.log_softmax(top_circuit_logit_c2h6)
            top_distribution_h7=F.log_softmax(top_circuit_logit_c2h7)
            top_distribution_h8=F.log_softmax(top_circuit_logit_c2h8)
            top_distribution_h9=F.log_softmax(top_circuit_logit_c2h9)
            top_distribution_h10=F.log_softmax(top_circuit_logit_c2h10)
            top_distribution_h11=F.log_softmax(top_circuit_logit_c2h11)
            top_distribution_h12=F.log_softmax(top_circuit_logit_c2h12)
            
            
            top_kld_c1=0.5*F.kl_div(top_distribution_c1,top_distribution_all_c1,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c1,reduction='sum')
            top_kld_c2=0.5*F.kl_div(top_distribution_c2,top_distribution_all_c2,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2,reduction='sum')
            top_kld_c3=0.5*F.kl_div(top_distribution_c3,top_distribution_all_c3,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c3,reduction='sum')
            top_kld_c4=0.5*F.kl_div(top_distribution_c4,top_distribution_all_c4,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c4,reduction='sum')
            top_kld_c5=0.5*F.kl_div(top_distribution_c5,top_distribution_all_c5,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c5,reduction='sum')
            top_kld_c6=0.5*F.kl_div(top_distribution_c6,top_distribution_all_c6,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c6,reduction='sum')
            top_kld_h1=0.5*F.kl_div(top_distribution_h1,top_distribution_all_c2h1,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h1,reduction='sum')
            top_kld_h2=0.5*F.kl_div(top_distribution_h2,top_distribution_all_c2h2,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h2,reduction='sum')
            top_kld_h3=0.5*F.kl_div(top_distribution_h3,top_distribution_all_c2h3,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h3,reduction='sum')
            top_kld_h4=0.5*F.kl_div(top_distribution_h4,top_distribution_all_c2h4,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h4,reduction='sum')
            top_kld_h5=0.5*F.kl_div(top_distribution_h5,top_distribution_all_c2h5,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h5,reduction='sum')
            top_kld_h6=0.5*F.kl_div(top_distribution_h6,top_distribution_all_c2h6,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h6,reduction='sum')
            top_kld_h7=0.5*F.kl_div(top_distribution_h7,top_distribution_all_c2h7,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h7,reduction='sum')
            top_kld_h8=0.5*F.kl_div(top_distribution_h8,top_distribution_all_c2h8,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h8,reduction='sum')
            top_kld_h9=0.5*F.kl_div(top_distribution_h9,top_distribution_all_c2h9,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h9,reduction='sum')
            top_kld_h10=0.5*F.kl_div(top_distribution_h10,top_distribution_all_c2h10,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h10,reduction='sum')
            top_kld_h11=0.5*F.kl_div(top_distribution_h11,top_distribution_all_c2h11,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h11,reduction='sum')
            top_kld_h12=0.5*F.kl_div(top_distribution_h12,top_distribution_all_c2h12,reduction='sum')+0.5*F.kl_div(top_distribution_all,top_distribution_all_c2h12,reduction='sum')
            top_head_list=[top_kld_h1,top_kld_h2,top_kld_h3,top_kld_h4,top_kld_h5,top_kld_h6,top_kld_h7,top_kld_h8,top_kld_h9,top_kld_h10,top_kld_h11,top_kld_h12]
            top_kld_min_head=top_head_list.index(min(top_head_list))
            
            return tokens_c1,tokens_c2,tokens_c2h1,tokens_c2h2,tokens_c2h3,tokens_c2h4,tokens_c2h5,tokens_c2h6,tokens_c2h7,tokens_c2h8,\
                tokens_c2h9,tokens_c2h10,tokens_c2h11,tokens_c2h12,tokens_c3,tokens_c4,tokens_c5,tokens_c6,\
cos_sim_c1,cos_sim_c2,cos_sim_c2h1,cos_sim_c2h2,cos_sim_c2h3,cos_sim_c2h4,cos_sim_c2h5,cos_sim_c2h6,cos_sim_c2h7,cos_sim_c2h8,cos_sim_c2h9,cos_sim_c2h10,\
cos_sim_c2h11,cos_sim_c2h12,cos_sim_c3,cos_sim_c4,cos_sim_c5,cos_sim_c6,\
    top_cos_sim_c1,top_cos_sim_c2,top_cos_sim_c2h1,top_cos_sim_c2h2,top_cos_sim_c2h3,top_cos_sim_c2h4,top_cos_sim_c2h5,top_cos_sim_c2h6,\
    top_cos_sim_c2h7,top_cos_sim_c2h8,top_cos_sim_c2h9,top_cos_sim_c2h10,top_cos_sim_c2h11,top_cos_sim_c2h12,top_cos_sim_c3,top_cos_sim_c4,\
        top_cos_sim_c5,top_cos_sim_c6,\
mse_c1,mse_c2,mse_c3,mse_c4,mse_c5,mse_c6,mse_c2h1,mse_c2h2,mse_c2h3,mse_c2h4,mse_c2h5,mse_c2h6,mse_c2h7,mse_c2h8,mse_c2h9,mse_c2h10,mse_c2h11,mse_c2h12,\
top_mse_c1,top_mse_c2,top_mse_c3,top_mse_c4,top_mse_c5,top_mse_c6,top_mse_c2h1,top_mse_c2h2,top_mse_c2h3,top_mse_c2h4,top_mse_c2h5,top_mse_c2h6,\
    top_mse_c2h7,top_mse_c2h8,top_mse_c2h9,top_mse_c2h10,top_mse_c2h11,top_mse_c2h12,\
ce_c1,ce_c2,ce_c3,ce_c4,ce_c5,ce_c6,ce_c2h1,ce_c2h2,ce_c2h3,ce_c2h4,ce_c2h5,ce_c2h6,ce_c2h7,ce_c2h8,ce_c2h9,ce_c2h10,ce_c2h11,ce_c2h12,\
top_ce_c1,top_ce_c2,top_ce_c3,top_ce_c4,top_ce_c5,top_ce_c6,top_ce_c2h1,top_ce_c2h2,top_ce_c2h3,top_ce_c2h4,top_ce_c2h5,top_ce_c2h6,\
    top_ce_c2h7,top_ce_c2h8,top_ce_c2h9,top_ce_c2h10,top_ce_c2h11,top_ce_c2h12,\
kld_c1,kld_c2,kld_c3,kld_c4,kld_c5,kld_c6,kld_h1,kld_h2,kld_h3,kld_h4,kld_h5,kld_h6,kld_h7,kld_h8,kld_h9,kld_h10,kld_h11,kld_h12,\
top_kld_c1,top_kld_c2,top_kld_c3,top_kld_c4,top_kld_c5,top_kld_c6,top_kld_h1,top_kld_h2,top_kld_h3,top_kld_h4,top_kld_h5,top_kld_h6,top_kld_h7,\
    top_kld_h8,top_kld_h9,top_kld_h10,top_kld_h11,top_kld_h12, \
        kld_min_head,top_kld_min_head