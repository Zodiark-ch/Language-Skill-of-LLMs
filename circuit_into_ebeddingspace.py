import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoTokenizer,GPT2LMHeadModel
from tqdm import trange
import numpy as np
import logging



class attention_circuit_check(nn.Module):
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


        
    
            
    def forward(self,inputs,input_text,word_idx,IO,IO_m1,IO_a1,S,S_m1,S_a1,S2,end):
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
        induction_weight2=torch.zeros((len(self.layers),12))
        previous_weight=torch.zeros((len(self.layers),12))
        previous_weight2=torch.zeros((len(self.layers),12))
        Name_weight=torch.zeros((len(self.layers),12))
        Name_weight2=torch.zeros((len(self.layers),12))
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
            induction_weight[i][0]=head1_weight[0][S2+1][S_a1]
            induction_weight[i][1]=head2_weight[0][S2+1][S_a1]
            induction_weight[i][2]=head3_weight[0][S2+1][S_a1]
            induction_weight[i][3]=head4_weight[0][S2+1][S_a1]
            induction_weight[i][4]=head5_weight[0][S2+1][S_a1]
            induction_weight[i][5]=head6_weight[0][S2+1][S_a1]
            induction_weight[i][6]=head7_weight[0][S2+1][S_a1]
            induction_weight[i][7]=head8_weight[0][S2+1][S_a1]
            induction_weight[i][8]=head9_weight[0][S2+1][S_a1]
            induction_weight[i][9]=head10_weight[0][S2+1][S_a1]
            induction_weight[i][10]=head11_weight[0][S2+1][S_a1]
            induction_weight[i][11]=head12_weight[0][S2+1][S_a1]
            
            induction_weight2[i][0]=head1_weight[0][S2+1][IO_a1]
            induction_weight2[i][1]=head2_weight[0][S2+1][IO_a1]
            induction_weight2[i][2]=head3_weight[0][S2+1][IO_a1]
            induction_weight2[i][3]=head4_weight[0][S2+1][IO_a1]
            induction_weight2[i][4]=head5_weight[0][S2+1][IO_a1]
            induction_weight2[i][5]=head6_weight[0][S2+1][IO_a1]
            induction_weight2[i][6]=head7_weight[0][S2+1][IO_a1]
            induction_weight2[i][7]=head8_weight[0][S2+1][IO_a1]
            induction_weight2[i][8]=head9_weight[0][S2+1][IO_a1]
            induction_weight2[i][9]=head10_weight[0][S2+1][IO_a1]
            induction_weight2[i][10]=head11_weight[0][S2+1][IO_a1]
            induction_weight2[i][11]=head12_weight[0][S2+1][IO_a1]
            
            #find previous heads
            previous_weight[i][0]=head1_weight[0][S_a1][S]
            previous_weight[i][1]=head2_weight[0][S_a1][S]
            previous_weight[i][2]=head3_weight[0][S_a1][S]
            previous_weight[i][3]=head4_weight[0][S_a1][S]
            previous_weight[i][4]=head5_weight[0][S_a1][S]
            previous_weight[i][5]=head6_weight[0][S_a1][S]
            previous_weight[i][6]=head7_weight[0][S_a1][S]
            previous_weight[i][7]=head8_weight[0][S_a1][S]
            previous_weight[i][8]=head9_weight[0][S_a1][S]
            previous_weight[i][9]=head10_weight[0][S_a1][S]
            previous_weight[i][10]=head11_weight[0][S_a1][S]
            previous_weight[i][11]=head12_weight[0][S_a1][S]
            
            previous_weight2[i][0]=head1_weight[0][IO_a1][IO]
            previous_weight2[i][1]=head2_weight[0][IO_a1][IO]
            previous_weight2[i][2]=head3_weight[0][IO_a1][IO]
            previous_weight2[i][3]=head4_weight[0][IO_a1][IO]
            previous_weight2[i][4]=head5_weight[0][IO_a1][IO]
            previous_weight2[i][5]=head6_weight[0][IO_a1][IO]
            previous_weight2[i][6]=head7_weight[0][IO_a1][IO]
            previous_weight2[i][7]=head8_weight[0][IO_a1][IO]
            previous_weight2[i][8]=head9_weight[0][IO_a1][IO]
            previous_weight2[i][9]=head10_weight[0][IO_a1][IO]
            previous_weight2[i][10]=head11_weight[0][IO_a1][IO]
            previous_weight2[i][11]=head12_weight[0][IO_a1][IO]
            
            #find name heads
            Name_weight[i][0]=head1_weight[0][end][S2]
            Name_weight[i][1]=head2_weight[0][end][S2]
            Name_weight[i][2]=head3_weight[0][end][S2]
            Name_weight[i][3]=head4_weight[0][end][S2]
            Name_weight[i][4]=head5_weight[0][end][S2]
            Name_weight[i][5]=head6_weight[0][end][S2]
            Name_weight[i][6]=head7_weight[0][end][S2]
            Name_weight[i][7]=head8_weight[0][end][S2]
            Name_weight[i][8]=head9_weight[0][end][S2]
            Name_weight[i][9]=head10_weight[0][end][S2]
            Name_weight[i][10]=head11_weight[0][end][S2]
            Name_weight[i][11]=head12_weight[0][end][S2]
            
            #find name heads
            Name_weight2[i][0]=head1_weight[0][end][IO]
            Name_weight2[i][1]=head2_weight[0][end][IO]
            Name_weight2[i][2]=head3_weight[0][end][IO]
            Name_weight2[i][3]=head4_weight[0][end][IO]
            Name_weight2[i][4]=head5_weight[0][end][IO]
            Name_weight2[i][5]=head6_weight[0][end][IO]
            Name_weight2[i][6]=head7_weight[0][end][IO]
            Name_weight2[i][7]=head8_weight[0][end][IO]
            Name_weight2[i][8]=head9_weight[0][end][IO]
            Name_weight2[i][9]=head10_weight[0][end][IO]
            Name_weight2[i][10]=head11_weight[0][end][IO]
            Name_weight2[i][11]=head12_weight[0][end][IO]
            
            
            
        return duplicate_weight,induction_weight,induction_weight2,previous_weight,previous_weight2,Name_weight,Name_weight2
            
            
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
        if self.args.logs=='true':
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
            if self.args.logs=='true':
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
            
            
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            if self.args.logs=='true':
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
            
            
            if self.args.logs=='true':
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
        
        
        if self.args.logs=='true':    
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
            top_circuit_logit,top_circuit_idx=torch.topk(circuit_logits,1)
            
            
            
            
            
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
        
        
        
        
class tokens_extraction(nn.Module):
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
        
        
        top_token_matrix=torch.zeros(12,10)#top 10 tokens 
        top_token_alltokens=torch.zeros(12,hidden_states.size(-2),1)
        
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
            

            
            #get top token_idx of circuit 1 in each layer:
            
            ln_hidden_state=self.model.transformer.ln_f(circuit_input)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,10)
            top_token_matrix[i]=predicted_indices[0]
            
            circuit_logits_alltokens=self.model.lm_head(ln_hidden_state)[0]
            _,predicted_indices_alltokens=torch.topk(circuit_logits_alltokens,1)
            top_token_alltokens[i]=predicted_indices_alltokens
        top_token_alltokens=top_token_alltokens.permute(2,1,0).int()
        
        token_sequence_all=[]
        for i in range(top_token_alltokens.size()[-2]):
            token_sequence_all.append(self.get_tokens(top_token_alltokens[0][i]))
        return top_token_matrix,top_token_alltokens.permute(2,1,0),token_sequence_all
            
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def get_tokens(self,predicted_indices):
        token_list=[]
        for i in range(12):
            ids=predicted_indices[i]
            token_list.append(self.tokenizer.decode(ids))
        return token_list                   
                       



class residual_analysis(nn.Module):
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


        
    
            
    def forward(self,inputs,top_token_matrix):
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
        initial_token,emerge_token,predicted_token=self.token_split(top_token_matrix)
        initial_token_recorder=torch.zeros((12,initial_token.size()[0]))
        emerge_token_recorder=torch.zeros((12,emerge_token.size()[0]))
        predicted_token_recorder=torch.zeros((12,predicted_token.size()[0]))
        
        
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
            
            #circuit_1 is the self path, only include itself
            circuit_1=circuit_input


            #get top token_idx of circuit 1 in each layer:
            
            ln_hidden_state=self.model.transformer.ln_f(circuit_1)
            circuit_logits=self.model.lm_head(ln_hidden_state)[0][-1].unsqueeze(0).cpu()
            initial_token_recorder[i]=circuit_logits.index_select(1,initial_token)
            emerge_token_recorder[i]=circuit_logits.index_select(1,emerge_token)
            predicted_token_recorder[i]=circuit_logits.index_select(1, predicted_token)
        initial_token_recorder=initial_token_recorder.transpose(0,1)
        emerge_token_recorder=emerge_token_recorder.transpose(0,1)
        predicted_token_recorder=predicted_token_recorder.transpose(0,1)
            
        return initial_token,emerge_token,predicted_token,initial_token_recorder.numpy(),emerge_token_recorder.numpy(),predicted_token_recorder.numpy()
            
            
        
            
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def token_split(self,top_token_matrix):
        initial_token=top_token_matrix[0].int()
        predicted_token=top_token_matrix[-1].int()
        all_token=top_token_matrix[1:-1].view(-1)
        emerge_token=torch.tensor([999999999])
        for i in range(all_token.size()[0]):
            if torch.any(initial_token.eq(all_token[i])).item() or torch.any(predicted_token.eq(all_token[i])).item() or torch.any(emerge_token.eq(all_token[i])).item(): 
                continue
            else:
                emerge_token=torch.cat((emerge_token,all_token[i:i+1]))
        return initial_token,emerge_token[1:].int(),predicted_token
                       
                       
class bias_analysis(nn.Module):
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


        
    
            
    def forward(self):
        
        past_key_values = tuple([None] * len(self.layers))
        
        
        top_token_matrix=torch.zeros(12,50)#top 10 tokens 
        top_token=[]
        top_token_logits=torch.zeros(12,50)
        
        top_attn_matrix=torch.zeros(12,50)#top 10 tokens 
        top_attn_token=[]
        top_attn_logits=torch.zeros(12,50)
        
        top_mlp_matrix=torch.zeros(12,50)#top 10 tokens 
        top_mlp_token=[]
        top_mlp_logits=torch.zeros(12,50)
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            
            W_obias=block.attn.c_proj.bias#R^[d]=[768],but in practice, we used R=[N,768]
            
            W_mlp2bias=block.mlp.c_proj.bias #R^[d]=[768] 
            
            #circuit_1 is the self path, only include itself
            circuit_6=W_obias+W_mlp2bias


            #get top token_idx of circuit 1 in each layer:
            
            ln_hidden_state=self.model.transformer.ln_f(circuit_6)
            circuit_logits=self.model.lm_head(ln_hidden_state).unsqueeze(0)
            _,predicted_indices=torch.topk(circuit_logits,50)
            top_token_matrix[i]=predicted_indices[0]
            top_token.append(self.get_tokens(predicted_indices[0]))
            top_token_logits[i]=circuit_logits.index_select(1,predicted_indices[0])
            
            ln_hidden_state_attn=self.model.transformer.ln_f(W_obias)
            circuit_logits_attn=self.model.lm_head(ln_hidden_state_attn).unsqueeze(0)
            _,predicted_indices_attn=torch.topk(circuit_logits_attn,50)
            top_attn_matrix[i]=predicted_indices_attn[0]
            top_attn_token.append(self.get_tokens(predicted_indices_attn[0]))
            top_attn_logits[i]=circuit_logits_attn.index_select(1,predicted_indices_attn[0])
            
            ln_hidden_state_mlp=self.model.transformer.ln_f(W_mlp2bias)
            circuit_logits_mlp=self.model.lm_head(ln_hidden_state_mlp).unsqueeze(0)
            _,predicted_indices_mlp=torch.topk(circuit_logits_mlp,50)
            top_mlp_matrix[i]=predicted_indices_mlp[0]
            top_mlp_token.append(self.get_tokens(predicted_indices_mlp[0]))
            top_mlp_logits[i]=circuit_logits_mlp.index_select(1,predicted_indices_mlp[0])
            
            
            
        return top_token_matrix,top_token,top_token_logits,top_attn_matrix,top_attn_token,top_attn_logits,top_mlp_matrix,top_mlp_token,top_mlp_logits
            
            
        
            
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def token_split(self,top_token_matrix):
        initial_token=top_token_matrix[0].int()
        predicted_token=top_token_matrix[-1].int()
        all_token=top_token_matrix[1:-1].view(-1)
        emerge_token=torch.tensor([999999999])
        for i in range(all_token.size()[0]):
            if torch.any(initial_token.eq(all_token[i])).item() or torch.any(predicted_token.eq(all_token[i])).item() or torch.any(emerge_token.eq(all_token[i])).item(): 
                continue
            else:
                emerge_token=torch.cat((emerge_token,all_token[i:i+1]))
        return initial_token,emerge_token[1:].int(),predicted_token
    
    def get_tokens(self,predicted_indices):
        token_list=[]
        for i in range(50):
            ids=predicted_indices[i]
            token_list.append(self.tokenizer.decode(ids))
        return token_list  
    
    


class attention_analysis(nn.Module):
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


        
    
            
    def forward(self,inputs,label_ids):
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
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
        if self.args.logs=='true':
            logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+self.tokenizer.decode(input_ids[0])+'_logging.log')
            logger.info('max probability tokens are:'+ self.tokenizer.decode(label_ids)+'with ids {}'.format(label_ids))
        attention_weight_alllayer=torch.zeros((12,12,input_ids.size()[-1],input_ids.size()[-1]))
        circuit_1_label_logits=torch.zeros((12,input_ids.size()[-1],label_ids.size()[-1]))
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
            circuit_1_logits_all=self.model.lm_head(ln_hidden_state_c1)[0]#[N,E]
            circuit_1_label_logits[i]=F.softmax(circuit_1_logits_all,dim=-1).index_select(-1,label_ids)#[12,N,1],12 represents layers
            
            #for each head, let A*X be the information passing and Wov*E be the memory vocabulary distribution
            # if the logits of label after A*X*Wov*E more than A*X, the knowledge inspires
            
            # get head_weight 
            AX_logits=torch.matmul(attn_weights,circuit_1_logits_all)#[12,N,E],12 represents the head nums
            AX_label_logits=F.softmax(AX_logits,dim=-1).index_select(-1,label_ids)#[12,N,1],12 represents heads nums
            AX_h1_logits,AX_h2_logits,AX_h3_logits,AX_h4_logits,AX_h5_logits,AX_h6_logits,AX_h7_logits,AX_h8_logits,AX_h9_logits,\
                AX_h10_logits,AX_h11_logits,AX_h12_logits=AX_label_logits.split(1,dim=0)#[1,N,1]
            
            #the circuit 2
            ln_hidden_state_c2=self.model.transformer.ln_f(circuit_2)
            circuit_2_logits=self.model.lm_head(ln_hidden_state_c2)[0][-1].unsqueeze(0)
            
            
            #get the output without biasWv 
            Output_mh_wobv=torch.matmul(attn_weights,Output_mhov)
            head1_attn_wobv,head2_attn_wobv,head3_attn_wobv,head4_attn_wobv,head5_attn_wobv,head6_attn_wobv,head7_attn_wobv,head8_attn_wobv,\
                head9_attn_wobv,head10_attn_wobv,head11_attn_wobv,head12_attn_wobv=Output_mh_wobv.split(1,dim=0)
            #each head
            ln_hidden_state_h1=self.model.transformer.ln_f(head1_attn)
            head1_logits=self.model.lm_head(ln_hidden_state_h1)[0].unsqueeze(0)
            head1_label_logits=F.softmax(head1_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h1=torch.where(head1_label_logits>AX_h1_logits,1,0)
            
            ln_hidden_state_h1_wobv=self.model.transformer.ln_f(head1_attn_wobv)
            head1_logits_wobv=self.model.lm_head(ln_hidden_state_h1_wobv)[0].unsqueeze(0)
            head1_label_logits_wobv=F.softmax(head1_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h1_wobv=torch.where(head1_label_logits_wobv>AX_h1_logits,1,0)
            
            
            #head 2
            ln_hidden_state_h2=self.model.transformer.ln_f(head2_attn)
            head2_logits=self.model.lm_head(ln_hidden_state_h2)[0].unsqueeze(0)
            head2_label_logits=F.softmax(head2_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h2=torch.where(head2_label_logits>AX_h2_logits,1,0)
            
            ln_hidden_state_h2_wobv=self.model.transformer.ln_f(head2_attn_wobv)
            head2_logits_wobv=self.model.lm_head(ln_hidden_state_h2_wobv)[0].unsqueeze(0)
            head2_label_logits_wobv=F.softmax(head2_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h2_wobv=torch.where(head2_label_logits_wobv>AX_h2_logits,1,0)
            
            
            #head 3
            ln_hidden_state_h3=self.model.transformer.ln_f(head3_attn)
            head3_logits=self.model.lm_head(ln_hidden_state_h3)[0].unsqueeze(0)
            head3_label_logits=F.softmax(head3_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h3=torch.where(head3_label_logits>AX_h3_logits,1,0)
            
            ln_hidden_state_h3_wobv=self.model.transformer.ln_f(head3_attn_wobv)
            head3_logits_wobv=self.model.lm_head(ln_hidden_state_h3_wobv)[0].unsqueeze(0)
            head3_label_logits_wobv=F.softmax(head3_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h3_wobv=torch.where(head3_label_logits_wobv>AX_h3_logits,1,0)
            
            
            #head 4
            ln_hidden_state_h4=self.model.transformer.ln_f(head4_attn)
            head4_logits=self.model.lm_head(ln_hidden_state_h4)[0].unsqueeze(0)
            head4_label_logits=F.softmax(head4_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h4=torch.where(head4_label_logits>AX_h4_logits,1,0)
            
            ln_hidden_state_h4_wobv=self.model.transformer.ln_f(head4_attn_wobv)
            head4_logits_wobv=self.model.lm_head(ln_hidden_state_h4_wobv)[0].unsqueeze(0)
            head4_label_logits_wobv=F.softmax(head4_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h4_wobv=torch.where(head4_label_logits_wobv>AX_h4_logits,1,0)
            
            
            #head 5
            ln_hidden_state_h5=self.model.transformer.ln_f(head5_attn)
            head5_logits=self.model.lm_head(ln_hidden_state_h5)[0].unsqueeze(0)
            head5_label_logits=F.softmax(head5_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h5=torch.where(head5_label_logits>AX_h5_logits,1,0)
            
            ln_hidden_state_h5_wobv=self.model.transformer.ln_f(head5_attn_wobv)
            head5_logits_wobv=self.model.lm_head(ln_hidden_state_h5_wobv)[0].unsqueeze(0)
            head5_label_logits_wobv=F.softmax(head5_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h5_wobv=torch.where(head5_label_logits_wobv>AX_h5_logits,1,0)
            
            
            #head 6
            ln_hidden_state_h6=self.model.transformer.ln_f(head6_attn)
            head6_logits=self.model.lm_head(ln_hidden_state_h6)[0].unsqueeze(0)
            head6_label_logits=F.softmax(head6_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h6=torch.where(head6_label_logits>AX_h6_logits,1,0)
            
            ln_hidden_state_h6_wobv=self.model.transformer.ln_f(head6_attn_wobv)
            head6_logits_wobv=self.model.lm_head(ln_hidden_state_h6_wobv)[0].unsqueeze(0)
            head6_label_logits_wobv=F.softmax(head6_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h6_wobv=torch.where(head6_label_logits_wobv>AX_h6_logits,1,0)
            
            
            #head 7
            ln_hidden_state_h7=self.model.transformer.ln_f(head7_attn)
            head7_logits=self.model.lm_head(ln_hidden_state_h7)[0].unsqueeze(0)
            head7_label_logits=F.softmax(head7_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h7=torch.where(head7_label_logits>AX_h7_logits,1,0)
            
            ln_hidden_state_h7_wobv=self.model.transformer.ln_f(head7_attn_wobv)
            head7_logits_wobv=self.model.lm_head(ln_hidden_state_h7_wobv)[0].unsqueeze(0)
            head7_label_logits_wobv=F.softmax(head7_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h7_wobv=torch.where(head7_label_logits_wobv>AX_h7_logits,1,0)
            
            
            #head 8
            ln_hidden_state_h8=self.model.transformer.ln_f(head8_attn)
            head8_logits=self.model.lm_head(ln_hidden_state_h8)[0].unsqueeze(0)
            head8_label_logits=F.softmax(head8_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h8=torch.where(head8_label_logits>AX_h8_logits,1,0)
            
            ln_hidden_state_h8_wobv=self.model.transformer.ln_f(head8_attn_wobv)
            head8_logits_wobv=self.model.lm_head(ln_hidden_state_h8_wobv)[0].unsqueeze(0)
            head8_label_logits_wobv=F.softmax(head8_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h8_wobv=torch.where(head8_label_logits_wobv>AX_h8_logits,1,0)
            
            
            #head 9
            ln_hidden_state_h9=self.model.transformer.ln_f(head9_attn)
            head9_logits=self.model.lm_head(ln_hidden_state_h9)[0].unsqueeze(0)
            head9_label_logits=F.softmax(head9_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h9=torch.where(head9_label_logits>AX_h9_logits,1,0)
            
            ln_hidden_state_h9_wobv=self.model.transformer.ln_f(head9_attn_wobv)
            head9_logits_wobv=self.model.lm_head(ln_hidden_state_h9_wobv)[0].unsqueeze(0)
            head9_label_logits_wobv=F.softmax(head9_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h9_wobv=torch.where(head9_label_logits_wobv>AX_h9_logits,1,0)
            
            
            #head 10
            ln_hidden_state_h10=self.model.transformer.ln_f(head10_attn)
            head10_logits=self.model.lm_head(ln_hidden_state_h10)[0].unsqueeze(0)
            head10_label_logits=F.softmax(head10_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h10=torch.where(head10_label_logits>AX_h10_logits,1,0)
            
            ln_hidden_state_h10_wobv=self.model.transformer.ln_f(head10_attn_wobv)
            head10_logits_wobv=self.model.lm_head(ln_hidden_state_h10_wobv)[0].unsqueeze(0)
            head10_label_logits_wobv=F.softmax(head10_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h10_wobv=torch.where(head10_label_logits_wobv>AX_h10_logits,1,0)
            
            
            #head 11
            ln_hidden_state_h11=self.model.transformer.ln_f(head11_attn)
            head11_logits=self.model.lm_head(ln_hidden_state_h11)[0].unsqueeze(0)
            head11_label_logits=F.softmax(head11_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h11=torch.where(head11_label_logits>AX_h11_logits,1,0)
            
            ln_hidden_state_h11_wobv=self.model.transformer.ln_f(head11_attn_wobv)
            head11_logits_wobv=self.model.lm_head(ln_hidden_state_h11_wobv)[0].unsqueeze(0)
            head11_label_logits_wobv=F.softmax(head11_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h11_wobv=torch.where(head11_label_logits_wobv>AX_h11_logits,1,0)
            
            
            #head 12
            ln_hidden_state_h12=self.model.transformer.ln_f(head12_attn)
            head12_logits=self.model.lm_head(ln_hidden_state_h12)[0].unsqueeze(0)
            head12_label_logits=F.softmax(head12_logits,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h12=torch.where(head12_label_logits>AX_h12_logits,1,0)
            
            ln_hidden_state_h12_wobv=self.model.transformer.ln_f(head12_attn_wobv)
            head12_logits_wobv=self.model.lm_head(ln_hidden_state_h12_wobv)[0].unsqueeze(0)
            head12_label_logits_wobv=F.softmax(head12_logits_wobv,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_h12_wobv=torch.where(head12_label_logits_wobv>AX_h12_logits,1,0)
            
            if self.args.logs=='true':
                logger.info('################{}-th layer#################'.format(i))
                
                logger.info('##{}-th layer ##Inspire##: The head1 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h1, Inspire_h1_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head2 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h2, Inspire_h2_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head3 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h3, Inspire_h3_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head4 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h4, Inspire_h4_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head5 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h5, Inspire_h5_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head6 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h6, Inspire_h6_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head7 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h7, Inspire_h7_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head8 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h8, Inspire_h8_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head9 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h9, Inspire_h9_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head10 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h10, Inspire_h10_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head11 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h11, Inspire_h11_wobv),dim=-1)))
                logger.info('##{}-th layer ##Inspire##: The head12 Inspire status of source tokens is \n {}'.format(i,torch.cat((Inspire_h12, Inspire_h12_wobv),dim=-1)))
               
            
            
            #show the attention weights 
            token_list=self.tokenizer.decode(input_ids[0])
            head1_weight, head2_weight,head3_weight, head4_weight,head5_weight, head6_weight,head7_weight, head8_weight,\
                head9_weight, head10_weight,head11_weight, head12_weight=attn_weights.split(1,dim=0)#[1,N,N]
            attention_weight_alllayer[i]=attn_weights
            if self.args.logs=='true':
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
        logging.shutdown()
        return attention_weight_alllayer        
            
            
            
            
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
    
    
class mlp_analysis(nn.Module):
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


        
    
            
    def forward(self,inputs,label_ids):
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
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
        if self.args.logs=='true':
            logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+self.tokenizer.decode(input_ids[0])+'_logging.log')
            logger.info('max probability tokens are:'+ self.tokenizer.decode(label_ids)+'with ids {}'.format(label_ids))
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
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit4_input_ln = block.ln_2(circuit_2)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            #get subcircuit of each head and compensation circuit
            #head1
            head1_c4in=block.ln_2(head1_attn)
            Output_mlp1_h1=torch.matmul(head1_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h1=Act_mlp(Output_mlp1_h1) #activated
            circuit_4_h1=torch.matmul(Output_mlp1_act_4_h1,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head2
            head2_c4in=block.ln_2(head2_attn)
            Output_mlp1_h2=torch.matmul(head2_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h2=Act_mlp(Output_mlp1_h2) #activated
            circuit_4_h2=torch.matmul(Output_mlp1_act_4_h2,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head3 
            head3_c4in=block.ln_2(head3_attn)
            Output_mlp1_h3=torch.matmul(head3_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h3=Act_mlp(Output_mlp1_h3) #activated
            circuit_4_h3=torch.matmul(Output_mlp1_act_4_h3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head4
            head4_c4in=block.ln_2(head4_attn)
            Output_mlp1_h4=torch.matmul(head4_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h4=Act_mlp(Output_mlp1_h4) #activated
            circuit_4_h4=torch.matmul(Output_mlp1_act_4_h4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
            
            #head5
            head5_c4in=block.ln_2(head5_attn)
            Output_mlp1_h5=torch.matmul(head5_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h5=Act_mlp(Output_mlp1_h5) #activated
            circuit_4_h5=torch.matmul(Output_mlp1_act_4_h5,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head6
            head6_c4in=block.ln_2(head6_attn)
            Output_mlp1_h6=torch.matmul(head6_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h6=Act_mlp(Output_mlp1_h6) #activated
            circuit_4_h6=torch.matmul(Output_mlp1_act_4_h6,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head7
            head7_c4in=block.ln_2(head7_attn)
            Output_mlp1_h7=torch.matmul(head7_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h7=Act_mlp(Output_mlp1_h7) #activated
            circuit_4_h7=torch.matmul(Output_mlp1_act_4_h7,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head8
            head8_c4in=block.ln_2(head8_attn)
            Output_mlp1_h8=torch.matmul(head8_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h8=Act_mlp(Output_mlp1_h8) #activated
            circuit_4_h8=torch.matmul(Output_mlp1_act_4_h8,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head9
            head9_c4in=block.ln_2(head9_attn)
            Output_mlp1_h9=torch.matmul(head9_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h9=Act_mlp(Output_mlp1_h9) #activated
            circuit_4_h9=torch.matmul(Output_mlp1_act_4_h9,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head10
            head10_c4in=block.ln_2(head10_attn)
            Output_mlp1_h10=torch.matmul(head10_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h10=Act_mlp(Output_mlp1_h10) #activated
            circuit_4_h10=torch.matmul(Output_mlp1_act_4_h10,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head11
            head11_c4in=block.ln_2(head11_attn)
            Output_mlp1_h11=torch.matmul(head11_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h11=Act_mlp(Output_mlp1_h11) #activated
            circuit_4_h11=torch.matmul(Output_mlp1_act_4_h11,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head12
            head12_c4in=block.ln_2(head12_attn)
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
            
            #get circuit sum 
            circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            
            
            #get Inspiration of each cirucit
            #C3 inspiration
            Inspire_c3=self.get_inspiration(circuit3_input_ln,circuit_3,label_ids)
            
            #C4 inspiration 
            Inspire_c4=self.get_inspiration(circuit4_input_ln,circuit_4,label_ids)
            
            #c4h1 inspiration
            Inspire_c4h1=self.get_inspiration(head1_c4in,circuit_4_h1,label_ids)
            
            #c4h2 inspiration
            Inspire_c4h2=self.get_inspiration(head2_c4in,circuit_4_h2,label_ids)
            
            #c4h3 inspiration
            Inspire_c4h3=self.get_inspiration(head3_c4in,circuit_4_h3,label_ids)
            
            #c4h4 inspiration
            Inspire_c4h4=self.get_inspiration(head4_c4in,circuit_4_h4,label_ids)
            
            #c4h5 inspiration
            Inspire_c4h5=self.get_inspiration(head5_c4in,circuit_4_h5,label_ids)
            
            #c4h6 inspiration
            Inspire_c4h6=self.get_inspiration(head6_c4in,circuit_4_h6,label_ids)
            
            #c4h7 inspiration
            Inspire_c4h7=self.get_inspiration(head7_c4in,circuit_4_h7,label_ids)
            
            #c4h8 inspiration
            Inspire_c4h8=self.get_inspiration(head8_c4in,circuit_4_h8,label_ids)
            
            #c4h9 inspiration
            Inspire_c4h9=self.get_inspiration(head9_c4in,circuit_4_h9,label_ids)
            
            #c4h10 inspiration
            Inspire_c4h10=self.get_inspiration(head10_c4in,circuit_4_h10,label_ids)
            
            #c4h11 inspiration
            Inspire_c4h11=self.get_inspiration(head11_c4in,circuit_4_h11,label_ids)
            
            #c4h12 inspiration
            Inspire_c4h12=self.get_inspiration(head12_c4in,circuit_4_h12,label_ids)
            
            #c4compt inspiration
            logits_c4=self.get_logits(circuit4_input_ln)
            logits_c4_h1=self.get_logits(head1_c4in)
            logits_c4_h2=self.get_logits(head2_c4in)
            logits_c4_h3=self.get_logits(head3_c4in)
            logits_c4_h4=self.get_logits(head4_c4in)
            logits_c4_h5=self.get_logits(head5_c4in)
            logits_c4_h6=self.get_logits(head6_c4in)
            logits_c4_h7=self.get_logits(head7_c4in)
            logits_c4_h8=self.get_logits(head8_c4in)
            logits_c4_h9=self.get_logits(head9_c4in)
            logits_c4_h10=self.get_logits(head10_c4in)
            logits_c4_h11=self.get_logits(head11_c4in)
            logits_c4_h12=self.get_logits(head12_c4in)
            logits_headall=logits_c4_h1+logits_c4_h2+logits_c4_h3+logits_c4_h4+logits_c4_h5+logits_c4_h6+logits_c4_h7+logits_c4_h8+\
                logits_c4_h9+logits_c4_h10+logits_c4_h11+logits_c4_h12
                
            label_logits_c4=F.softmax(logits_c4,dim=-1).index_select(-1,label_ids)#[1,N,1]
            label_logit_headall=F.softmax(logits_headall,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_c4compt=torch.where(label_logits_c4>label_logit_headall,1,0)
            
            #circuit 5
            logits_stream_all=self.get_logits(circuit_stream_all)
            logits_c3=self.get_logits(circuit3_input_ln)
            Output_mlp1_bias=Act_mlp(W_mlp1bias) #activated
            circuit_uni_wmlp1bias=torch.matmul(Output_mlp1_bias,W_mlp2)#R^[B,N,d]=[1,14,768]
            logits_wmlp1bias=self.get_logits(circuit_uni_wmlp1bias)
            logits_sub_all=logits_c3+logits_c4+logits_wmlp1bias
            label_logits_stream_all=F.softmax(logits_stream_all,dim=-1).index_select(-1,label_ids)#[1,N,1]
            label_logits_sub_all=F.softmax(logits_sub_all,dim=-1).index_select(-1,label_ids)#[1,N,1]
            Inspire_c5=torch.where(label_logits_stream_all>label_logits_sub_all,1,0)
            
            
            
            if self.args.logs=='true':
                logger.info('################{}-th layer#################'.format(i))
                logger.info('##{}-th layer ##Inspire##: The circuit3 Inspire status of source tokens is \n {}'.format(i,Inspire_c3))
                logger.info('##{}-th layer ##Inspire##: The circuit4 Inspire status of source tokens is \n {}'.format(i,Inspire_c4))
                
                logger.info('##{}-th layer ##Inspire##: The c4head1 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h1))
                logger.info('##{}-th layer ##Inspire##: The c4head2 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h2))
                logger.info('##{}-th layer ##Inspire##: The c4head3 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h3))
                logger.info('##{}-th layer ##Inspire##: The c4head4 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h4))
                logger.info('##{}-th layer ##Inspire##: The c4head5 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h5))
                logger.info('##{}-th layer ##Inspire##: The c4head6 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h6))
                logger.info('##{}-th layer ##Inspire##: The c4head7 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h7))
                logger.info('##{}-th layer ##Inspire##: The c4head8 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h8))
                logger.info('##{}-th layer ##Inspire##: The c4head9 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h9))
                logger.info('##{}-th layer ##Inspire##: The c4head10 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h10))
                logger.info('##{}-th layer ##Inspire##: The c4head11 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h11))
                logger.info('##{}-th layer ##Inspire##: The c4head12 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h12))
        
                logger.info('##{}-th layer ##Inspire##: The Inspire_c4compt status of source tokens is \n {}'.format(i,Inspire_c4compt))
                logger.info('##{}-th layer ##Inspire##: The circuit5 Inspire status of source tokens is \n {}'.format(i,Inspire_c5))
        
        logging.shutdown()
            
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def get_inspiration(self,input,output,label_ids):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        label_logits_in=F.softmax(logits_in,dim=-1).index_select(-1,label_ids)#[1,N,1]
        
        ln_hidden_state=self.model.transformer.ln_f(output)
        logits_out=self.model.lm_head(ln_hidden_state)[0].unsqueeze(0)
        label_logits=F.softmax(logits_out,dim=-1).index_select(-1,label_ids)#[1,N,1]
        Inspire=torch.where(label_logits>label_logits_in,1,0)
        return Inspire
    
    def get_logits(self,input):
        ln_hidden_state_in=self.model.transformer.ln_f(input)
        logits_in=self.model.lm_head(ln_hidden_state_in)[0].unsqueeze(0)
        
        return logits_in
                       
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
    
    
    
    
class distribution_analysis(nn.Module):
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


        
    
            
    def forward(self):
        
        past_key_values = tuple([None] * len(self.layers))
        
        
        if self.args.logs=='true':
                logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+'logging.log')
        for i, (block, layer_past) in enumerate(zip(self.layers, past_key_values)):
            
            W_qkv=block.attn.c_attn.weight #R^[d,3a]=[768,2304]
            W_q,W_k,W_v=W_qkv.split(768, dim=1)#R^[d,a]=[768,768]
            W_mhv=self._split_heads(W_v,12,64)
            W_o=block.attn.c_proj.weight#R^[a,d]=[768,768]
            W_mho=self._split_heads(W_o.transpose(-1,-2),12,64).transpose(-1,-2)#because a is first dim, so need transpose, R^[H,a/H,D]=[12,64,768]
            W_mhov=torch.matmul(W_mhv,W_mho)#R^[H,d,d]=[12,768,768]
            
            W_mlp2=block.mlp.c_proj.weight #R^[m,d]=[3072,768]
            
            
            #get the Wov of each head 
            Wov_h1,Wov_h2,Wov_h3,Wov_h4,Wov_h5,Wov_h6,Wov_h7,Wov_h8,Wov_h9,Wov_h10,Wov_h11,Wov_h12=W_mhov.split(1,dim=0)
            
           #get each memory's distribution
           
            circuit_logits_h1,predicted_indices_h1=self.get_logits_top(Wov_h1)
            circuit_logits_h2,predicted_indices_h2=self.get_logits_top(Wov_h2)
            circuit_logits_h3,predicted_indices_h3=self.get_logits_top(Wov_h3)
            circuit_logits_h4,predicted_indices_h4=self.get_logits_top(Wov_h4)
            circuit_logits_h5,predicted_indices_h5=self.get_logits_top(Wov_h5)
            circuit_logits_h6,predicted_indices_h6=self.get_logits_top(Wov_h6)
            circuit_logits_h7,predicted_indices_h7=self.get_logits_top(Wov_h7)
            circuit_logits_h8,predicted_indices_h8=self.get_logits_top(Wov_h8)
            circuit_logits_h9,predicted_indices_h9=self.get_logits_top(Wov_h9)
            circuit_logits_h10,predicted_indices_h10=self.get_logits_top(Wov_h10)
            circuit_logits_h11,predicted_indices_h11=self.get_logits_top(Wov_h11)
            circuit_logits_h12,predicted_indices_h12=self.get_logits_top(Wov_h12)
            
            circuit_logits_mlp,predicted_indices_mlp=self.get_logits_top(W_mlp2)
            top_tokens_h1,top_tokens_h2,top_tokens_h3,top_tokens_h4,top_tokens_h5,top_tokens_h6,\
                top_tokens_h7,top_tokens_h8,top_tokens_h9,top_tokens_h10,top_tokens_h11,top_tokens_h12,top_tokens_mlp,=\
                    [],[],[],[],[],[],[],[],[],[],[],[],[]
                    
            for m in range(predicted_indices_h2.size()[-2]):
                top_tokens_h1.append(self.get_tokens(predicted_indices_h1[0][m]))
                top_tokens_h2.append(self.get_tokens(predicted_indices_h2[0][m]))
                top_tokens_h3.append(self.get_tokens(predicted_indices_h3[0][m]))
                top_tokens_h4.append(self.get_tokens(predicted_indices_h4[0][m]))
                top_tokens_h5.append(self.get_tokens(predicted_indices_h5[0][m]))
                top_tokens_h6.append(self.get_tokens(predicted_indices_h6[0][m]))
                top_tokens_h7.append(self.get_tokens(predicted_indices_h7[0][m]))
                top_tokens_h8.append(self.get_tokens(predicted_indices_h8[0][m]))
                top_tokens_h9.append(self.get_tokens(predicted_indices_h9[0][m]))
                top_tokens_h10.append(self.get_tokens(predicted_indices_h10[0][m]))
                top_tokens_h11.append(self.get_tokens(predicted_indices_h11[0][m]))
                top_tokens_h12.append(self.get_tokens(predicted_indices_h12[0][m]))
                
            for m in range(predicted_indices_mlp.size()[0]):
                top_tokens_mlp.append(self.get_tokens(predicted_indices_mlp[m]))
            
            if self.args.logs=='true':
                
                logger.info('########## For the {}-th layer##########'.format(i))
                
                logger.info('For Head1 logits: {}'.format(circuit_logits_h1[0][:10][:50]))
                logger.info('For Head2 logits: {}'.format(circuit_logits_h2[0][:10][:50]))
                logger.info('For Head3 logits: {}'.format(circuit_logits_h3[0][:10][:50]))
                logger.info('For Head4 logits: {}'.format(circuit_logits_h4[0][:10][:50]))
                logger.info('For Head5 logits: {}'.format(circuit_logits_h5[0][:10][:50]))
                logger.info('For Head6 logits: {}'.format(circuit_logits_h6[0][:10][:50]))
                logger.info('For Head7 logits: {}'.format(circuit_logits_h7[0][:10][:50]))
                logger.info('For Head8 logits: {}'.format(circuit_logits_h8[0][:10][:50]))
                logger.info('For Head9 logits: {}'.format(circuit_logits_h9[0][:10][:50]))
                logger.info('For Head10 logits: {}'.format(circuit_logits_h10[0][:10][:50]))
                logger.info('For Head11 logits: {}'.format(circuit_logits_h11[0][:10][:50]))
                logger.info('For Head12 logits: {}'.format(circuit_logits_h12[0][:10][:50]))
                logger.info('For MLP logits: {}'.format(circuit_logits_mlp[:10][:50]))
                
                logger.info('For Head1 token distribution: {}'.format(top_tokens_h1))
                logger.info('For Head2 token distribution: {}'.format(top_tokens_h2))
                logger.info('For Head3 token distribution: {}'.format(top_tokens_h3))
                logger.info('For Head4 token distribution: {}'.format(top_tokens_h4))
                logger.info('For Head5 token distribution: {}'.format(top_tokens_h5))
                logger.info('For Head6 token distribution: {}'.format(top_tokens_h6))
                logger.info('For Head7 token distribution: {}'.format(top_tokens_h7))
                logger.info('For Head8 token distribution: {}'.format(top_tokens_h8))
                logger.info('For Head9 token distribution: {}'.format(top_tokens_h9))
                logger.info('For Head10 token distribution: {}'.format(top_tokens_h10))
                logger.info('For Head11 token distribution: {}'.format(top_tokens_h11))
                logger.info('For Head12 token distribution: {}'.format(top_tokens_h12))
                logger.info('For MLP token distribution: {}'.format(top_tokens_mlp))
                
            
            
            
        
            
            
        
            
            
            
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (batch, head, seq_length, head_features)
    
    def token_split(self,top_token_matrix):
        initial_token=top_token_matrix[0].int()
        predicted_token=top_token_matrix[-1].int()
        all_token=top_token_matrix[1:-1].view(-1)
        emerge_token=torch.tensor([999999999])
        for i in range(all_token.size()[0]):
            if torch.any(initial_token.eq(all_token[i])).item() or torch.any(predicted_token.eq(all_token[i])).item() or torch.any(emerge_token.eq(all_token[i])).item(): 
                continue
            else:
                emerge_token=torch.cat((emerge_token,all_token[i:i+1]))
        return initial_token,emerge_token[1:].int(),predicted_token
    
    def get_tokens(self,predicted_indices):
        token_list=[]
        for i in range(50):
            ids=predicted_indices[i]
            token_list.append(self.tokenizer.decode(ids))
        return token_list  
    
    def get_logits_top(self,W):
        ln_hidden_state=self.model.transformer.ln_f(W)
        circuit_logits=self.model.lm_head(ln_hidden_state)
        _,predicted_indices=torch.topk(circuit_logits,50)
        return circuit_logits,predicted_indices
    
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
    
    
    
    
class satisfiability_analysis(nn.Module):
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


        
    
            
    def forward(self,inputs,label_ids):
        inputs=inputs.to(self.device)
        label_ids=label_ids.to(self.device)
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
        if self.args.logs=='true':
            logger = self.get_logger('logs/' +self.args.task_name+'/'+ self.args.model_name +'/'+self.tokenizer.decode(input_ids[0])+'_logging.log')
            logger.info('The tokens are {}'.format(self.get_tokens(input_ids[0])))
            logger.info('max probability tokens are:'+ self.tokenizer.decode(label_ids)+'with ids {}'.format(label_ids))
        attention_weight_alllayer=torch.zeros((12,12,input_ids.size()[-1],input_ids.size()[-1]))
        circuit_1_label_logits=torch.zeros((12,input_ids.size()[-1],label_ids.size()[-1]))
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
            residual_stream=circuit_1+circuit_2+W_obias
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
            Output_mlp1=torch.matmul(circuit3_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_3=Act_mlp(Output_mlp1) #activated
            circuit_3=torch.matmul(Output_mlp1_act_3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            
            
            #circuit_4 is the attention+mlp path, attention_weight is as the same as one in circuit_2, but OVpath differs 
            circuit4_input_ln = block.ln_2(circuit_2+W_obias)# make representation matrix get normed R^[N,d]=[14,768]
            Output_mlp1=torch.matmul(circuit4_input_ln,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4=Act_mlp(Output_mlp1) #activated
            circuit_4=torch.matmul(Output_mlp1_act_4,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            #get subcircuit of each head and compensation circuit
            #head1
            head1_c4in=block.ln_2(head1_attn)
            Output_mlp1_h1=torch.matmul(head1_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h1=Act_mlp(Output_mlp1_h1) #activated
            circuit_4_h1=torch.matmul(Output_mlp1_act_4_h1,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head2
            head2_c4in=block.ln_2(head2_attn)
            Output_mlp1_h2=torch.matmul(head2_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h2=Act_mlp(Output_mlp1_h2) #activated
            circuit_4_h2=torch.matmul(Output_mlp1_act_4_h2,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head3 
            head3_c4in=block.ln_2(head3_attn)
            Output_mlp1_h3=torch.matmul(head3_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h3=Act_mlp(Output_mlp1_h3) #activated
            circuit_4_h3=torch.matmul(Output_mlp1_act_4_h3,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head4
            head4_c4in=block.ln_2(head4_attn)
            Output_mlp1_h4=torch.matmul(head4_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h4=Act_mlp(Output_mlp1_h4) #activated
            circuit_4_h4=torch.matmul(Output_mlp1_act_4_h4,W_mlp2)#R^[B,N,d]=[1,14,768]
    
            
            #head5
            head5_c4in=block.ln_2(head5_attn)
            Output_mlp1_h5=torch.matmul(head5_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h5=Act_mlp(Output_mlp1_h5) #activated
            circuit_4_h5=torch.matmul(Output_mlp1_act_4_h5,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head6
            head6_c4in=block.ln_2(head6_attn)
            Output_mlp1_h6=torch.matmul(head6_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h6=Act_mlp(Output_mlp1_h6) #activated
            circuit_4_h6=torch.matmul(Output_mlp1_act_4_h6,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head7
            head7_c4in=block.ln_2(head7_attn)
            Output_mlp1_h7=torch.matmul(head7_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h7=Act_mlp(Output_mlp1_h7) #activated
            circuit_4_h7=torch.matmul(Output_mlp1_act_4_h7,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head8
            head8_c4in=block.ln_2(head8_attn)
            Output_mlp1_h8=torch.matmul(head8_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h8=Act_mlp(Output_mlp1_h8) #activated
            circuit_4_h8=torch.matmul(Output_mlp1_act_4_h8,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head9
            head9_c4in=block.ln_2(head9_attn)
            Output_mlp1_h9=torch.matmul(head9_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h9=Act_mlp(Output_mlp1_h9) #activated
            circuit_4_h9=torch.matmul(Output_mlp1_act_4_h9,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head10
            head10_c4in=block.ln_2(head10_attn)
            Output_mlp1_h10=torch.matmul(head10_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h10=Act_mlp(Output_mlp1_h10) #activated
            circuit_4_h10=torch.matmul(Output_mlp1_act_4_h10,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head11
            head11_c4in=block.ln_2(head11_attn)
            Output_mlp1_h11=torch.matmul(head11_c4in,W_mlp1) #R^[B,N,m]=[1,14,3072]
            Output_mlp1_act_4_h11=Act_mlp(Output_mlp1_h11) #activated
            circuit_4_h11=torch.matmul(Output_mlp1_act_4_h11,W_mlp2)#R^[B,N,d]=[1,14,768]
            
            
            #head12
            head12_c4in=block.ln_2(head12_attn)
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
            
            #get circuit sum 
            circuit_sum=circuit_1+circuit_2+circuit_3+circuit_4+circuit_5+circuit_6 #R^[B,N,D]=[1,14,768]
            
            #get the output without biasWv 
            Output_mh_wobv=torch.matmul(attn_weights,Output_mhov)
            head1_attn_wobv,head2_attn_wobv,head3_attn_wobv,head4_attn_wobv,head5_attn_wobv,head6_attn_wobv,head7_attn_wobv,head8_attn_wobv,\
                head9_attn_wobv,head10_attn_wobv,head11_attn_wobv,head12_attn_wobv=Output_mh_wobv.split(1,dim=0)
            #each head
            
            #the circuit 1
            ln_hidden_state_c1=self.model.transformer.ln_f(circuit_1)
            circuit_1_logits_all=self.model.lm_head(ln_hidden_state_c1)[0]#[N,E]
            circuit_1_label_logits[i]=F.softmax(circuit_1_logits_all,dim=-1).index_select(-1,label_ids)#[12,N,1],12 represents layers
            
            #for each head, let A*X be the information passing and Wov*E be the memory vocabulary distribution
            # if the logits of label after A*X*Wov*E more than A*X, the knowledge inspires
            
            # get head_weight 
            AX_logits=torch.matmul(attn_weights,circuit_1_logits_all)#[12,N,E],12 represents the head nums
            AX_h1_logits_wonorm,AX_h2_logits_wonorm,AX_h3_logits_wonorm,AX_h4_logits_wonorm,AX_h5_logits_wonorm,AX_h6_logits_wonorm,\
                AX_h7_logits_wonorm,AX_h8_logits_wonorm,AX_h9_logits_wonorm,AX_h10_logits_wonorm,AX_h11_logits_wonorm,AX_h12_logits_wonorm=\
                    AX_logits.split(1,dim=0)#[1,N,E]
            

            
            #head1 in circuit2
            logit_c2h1=self.get_logits(head1_attn_wobv)#[1,N,E]
            Inspire_c2h1=self.get_inspiration_num(logit_c2h1,AX_h1_logits_wonorm,label_ids)
            
            #head 2 in circuit2
            logit_c2h2=self.get_logits(head2_attn_wobv)#[1,N,E]
            Inspire_c2h2=self.get_inspiration_num(logit_c2h2,AX_h2_logits_wonorm,label_ids)
            
            #head 3 in circuit2
            logit_c2h3=self.get_logits(head3_attn_wobv)#[1,N,E]
            Inspire_c2h3=self.get_inspiration_num(logit_c2h3,AX_h3_logits_wonorm,label_ids)
            
            #head 4 in circuit2
            logit_c2h4=self.get_logits(head4_attn_wobv)#[1,N,E]
            Inspire_c2h4=self.get_inspiration_num(logit_c2h4,AX_h4_logits_wonorm,label_ids)
            
            #head 5 in circuit2
            logit_c2h5=self.get_logits(head5_attn_wobv)#[1,N,E]
            Inspire_c2h5=self.get_inspiration_num(logit_c2h5,AX_h5_logits_wonorm,label_ids)
            
            #head 6 in circuit2
            logit_c2h6=self.get_logits(head6_attn_wobv)#[1,N,E]
            Inspire_c2h6=self.get_inspiration_num(logit_c2h6,AX_h6_logits_wonorm,label_ids)
            
            #head 7 in circuit2
            logit_c2h7=self.get_logits(head7_attn_wobv)#[1,N,E]
            Inspire_c2h7=self.get_inspiration_num(logit_c2h7,AX_h7_logits_wonorm,label_ids)
            
            #head 8 in circuit2
            logit_c2h8=self.get_logits(head8_attn_wobv)#[1,N,E]
            Inspire_c2h8=self.get_inspiration_num(logit_c2h8,AX_h8_logits_wonorm,label_ids)
            
            #head 9 in circuit2
            logit_c2h9=self.get_logits(head9_attn_wobv)#[1,N,E]
            Inspire_c2h9=self.get_inspiration_num(logit_c2h9,AX_h9_logits_wonorm,label_ids)
            
            #head 10 in circuit2
            logit_c2h10=self.get_logits(head10_attn_wobv)#[1,N,E]
            Inspire_c2h10=self.get_inspiration_num(logit_c2h10,AX_h10_logits_wonorm,label_ids)
            
            #head 11 in circuit2
            logit_c2h11=self.get_logits(head11_attn_wobv)#[1,N,E]
            Inspire_c2h11=self.get_inspiration_num(logit_c2h11,AX_h11_logits_wonorm,label_ids)
            
            #head 12 in circuit2
            logit_c2h12=self.get_logits(head12_attn_wobv)#[1,N,E]
            Inspire_c2h12=self.get_inspiration_num(logit_c2h12,AX_h12_logits_wonorm,label_ids)
            
            
            
            #C3 inspiration
            logit_c3=self.get_logits(circuit_3)#[1,N,E]
            logit_c3_origin=self.get_logits(circuit3_input_ln)#[1,N,E]
            Inspire_c3=self.get_inspiration_num(logit_c3,logit_c3_origin,label_ids)
            
            #C4 MLP inspiration 
            logit_c4=self.get_logits(circuit_4)#[1,N,E]
            logit_c4_origin=self.get_logits(circuit4_input_ln)#[1,N,E]
            Inspire_c4=self.get_inspiration_num(logit_c4,logit_c4_origin,label_ids)
            
            #c4h1 inspiration
            logit_c4h1=self.get_logits(circuit_4_h1)#[1,N,E]
            logit_c4h1_origin=self.get_logits(head1_c4in)#[1,N,E]
            Inspire_c4h1=self.get_inspiration_num(logit_c4h1,logit_c4h1_origin,label_ids)
        
            #c4h2 inspiration
            logit_c4h2=self.get_logits(circuit_4_h2)#[1,N,E]
            logit_c4h2_origin=self.get_logits(head2_c4in)#[1,N,E]
            Inspire_c4h2=self.get_inspiration_num(logit_c4h2,logit_c4h2_origin,label_ids)
            
            #c4h3 inspiration
            logit_c4h3=self.get_logits(circuit_4_h3)#[1,N,E]
            logit_c4h3_origin=self.get_logits(head3_c4in)#[1,N,E]
            Inspire_c4h3=self.get_inspiration_num(logit_c4h3,logit_c4h3_origin,label_ids)
            
            #c4h4 inspiration
            logit_c4h4=self.get_logits(circuit_4_h4)#[1,N,E]
            logit_c4h4_origin=self.get_logits(head4_c4in)#[1,N,E]
            Inspire_c4h4=self.get_inspiration_num(logit_c4h4,logit_c4h4_origin,label_ids)
            
            #c4h5 inspiration
            logit_c4h5=self.get_logits(circuit_4_h5)#[1,N,E]
            logit_c4h5_origin=self.get_logits(head5_c4in)#[1,N,E]
            Inspire_c4h5=self.get_inspiration_num(logit_c4h5,logit_c4h5_origin,label_ids)
            
            #c4h6 inspiration
            logit_c4h6=self.get_logits(circuit_4_h6)#[1,N,E]
            logit_c4h6_origin=self.get_logits(head6_c4in)#[1,N,E]
            Inspire_c4h6=self.get_inspiration_num(logit_c4h6,logit_c4h6_origin,label_ids)
            
            #c4h7 inspiration
            logit_c4h7=self.get_logits(circuit_4_h7)#[1,N,E]
            logit_c4h7_origin=self.get_logits(head7_c4in)#[1,N,E]
            Inspire_c4h7=self.get_inspiration_num(logit_c4h7,logit_c4h7_origin,label_ids)
            
            #c4h8 inspiration
            logit_c4h8=self.get_logits(circuit_4_h8)#[1,N,E]
            logit_c4h8_origin=self.get_logits(head8_c4in)#[1,N,E]
            Inspire_c4h8=self.get_inspiration_num(logit_c4h8,logit_c4h8_origin,label_ids)
            
            #c4h9 inspiration
            logit_c4h9=self.get_logits(circuit_4_h9)#[1,N,E]
            logit_c4h9_origin=self.get_logits(head9_c4in)#[1,N,E]
            Inspire_c4h9=self.get_inspiration_num(logit_c4h9,logit_c4h9_origin,label_ids)
            
            #c4h10 inspiration
            logit_c4h10=self.get_logits(circuit_4_h10)#[1,N,E]
            logit_c4h10_origin=self.get_logits(head10_c4in)#[1,N,E]
            Inspire_c4h10=self.get_inspiration_num(logit_c4h10,logit_c4h10_origin,label_ids)
            
            #c4h11 inspiration
            logit_c4h11=self.get_logits(circuit_4_h11)#[1,N,E]
            logit_c4h11_origin=self.get_logits(head11_c4in)#[1,N,E]
            Inspire_c4h11=self.get_inspiration_num(logit_c4h11,logit_c4h11_origin,label_ids)
            
            #c4h12 inspiration
            logit_c4h12=self.get_logits(circuit_4_h12)#[1,N,E]
            logit_c4h12_origin=self.get_logits(head12_c4in)#[1,N,E]
            Inspire_c4h12=self.get_inspiration_num(logit_c4h12,logit_c4h12_origin,label_ids)
            
            
            #c4h1 inspiration of attn+MLP
            Inspire_c4h1_all=self.get_inspiration_num(logit_c4h1,AX_h1_logits_wonorm,label_ids)
        
            #c4h2 inspiration of attn+MLP
            Inspire_c4h2_all=self.get_inspiration_num(logit_c4h2,AX_h1_logits_wonorm,label_ids)
            
            #c4h3 inspiration of attn+MLP
            Inspire_c4h3_all=self.get_inspiration_num(logit_c4h3,AX_h1_logits_wonorm,label_ids)
            
            #c4h4 inspiration of attn+MLP
            Inspire_c4h4_all=self.get_inspiration_num(logit_c4h4,AX_h1_logits_wonorm,label_ids)
            
            #c4h5 inspiration of attn+MLP
            Inspire_c4h5_all=self.get_inspiration_num(logit_c4h5,AX_h1_logits_wonorm,label_ids)
            
            #c4h6 inspiration of attn+MLP
            Inspire_c4h6_all=self.get_inspiration_num(logit_c4h6,AX_h1_logits_wonorm,label_ids)
            
            #c4h7 inspiration of attn+MLP
            Inspire_c4h7_all=self.get_inspiration_num(logit_c4h7,AX_h1_logits_wonorm,label_ids)
            
            #c4h8 inspiration of attn+MLP
            Inspire_c4h8_all=self.get_inspiration_num(logit_c4h8,AX_h1_logits_wonorm,label_ids)
            
            #c4h9 inspiration of attn+MLP
            Inspire_c4h9_all=self.get_inspiration_num(logit_c4h9,AX_h1_logits_wonorm,label_ids)
            
            #c4h10 inspiration of attn+MLP
            Inspire_c4h10_all=self.get_inspiration_num(logit_c4h10,AX_h1_logits_wonorm,label_ids)
            
            #c4h11 inspiration of attn+MLP
            Inspire_c4h11_all=self.get_inspiration_num(logit_c4h11,AX_h1_logits_wonorm,label_ids)
            
            #c4h12 inspiration of attn+MLP
            Inspire_c4h12_all=self.get_inspiration_num(logit_c4h12,AX_h1_logits_wonorm,label_ids)
            
            #c4compt inspiration
            logits_c4=self.get_logits(circuit4_input_ln)
            logits_c4_h1=self.get_logits(head1_c4in)
            logits_c4_h2=self.get_logits(head2_c4in)
            logits_c4_h3=self.get_logits(head3_c4in)
            logits_c4_h4=self.get_logits(head4_c4in)
            logits_c4_h5=self.get_logits(head5_c4in)
            logits_c4_h6=self.get_logits(head6_c4in)
            logits_c4_h7=self.get_logits(head7_c4in)
            logits_c4_h8=self.get_logits(head8_c4in)
            logits_c4_h9=self.get_logits(head9_c4in)
            logits_c4_h10=self.get_logits(head10_c4in)
            logits_c4_h11=self.get_logits(head11_c4in)
            logits_c4_h12=self.get_logits(head12_c4in)
            logits_headall=logits_c4_h1+logits_c4_h2+logits_c4_h3+logits_c4_h4+logits_c4_h5+logits_c4_h6+logits_c4_h7+logits_c4_h8+\
                logits_c4_h9+logits_c4_h10+logits_c4_h11+logits_c4_h12
                
            
            Inspire_c4compt=self.get_inspiration_num(logit_c4,logits_headall,label_ids)
            
            #circuit 5
            logits_stream_all=self.get_logits(circuit_stream_all)
            logits_c3=self.get_logits(circuit3_input_ln)
            Output_mlp1_bias=Act_mlp(W_mlp1bias) #activated
            circuit_uni_wmlp1bias=torch.matmul(Output_mlp1_bias,W_mlp2)#R^[B,N,d]=[1,14,768]
            logits_wmlp1bias=self.get_logits(circuit_uni_wmlp1bias)
            logits_sub_all=logits_c3+logits_c4+logits_wmlp1bias
        
            Inspire_c5=self.get_inspiration_num(logits_stream_all,logits_sub_all,label_ids)
            
            #get each circuits' label rank
            logit_c1=self.get_logits(circuit_1)
            logit_c2=self.get_logits(circuit_2)
            
            logit_c5=self.get_logits(circuit_5)
            logit_c6=self.get_logits(circuit_6)
            logit_all=self.get_logits(circuit_sum)
            logit_hs=self.get_logits(hidden_states)
            lable_rank_c1=self.get_token_rank(logit_c1,label_ids)
            lable_rank_c2=self.get_token_rank(logit_c2,label_ids)
            lable_rank_c3=self.get_token_rank(logit_c3,label_ids)
            lable_rank_c4=self.get_token_rank(logit_c4,label_ids)
            lable_rank_c5=self.get_token_rank(logit_c5,label_ids)
            lable_rank_c6=self.get_token_rank(logit_c6,label_ids)
            lable_rank_sum=self.get_token_rank(logit_all,label_ids)
            label_rank_hs=self.get_token_rank(logit_hs,label_ids)
        
            if self.args.logs=='true':
                logger.info('################{}-th layer#################'.format(i))
                logger.info('####CIRCUIT RANK####')
                logger.info('The Circuit 1 has label_rank \n {}'.format(lable_rank_c1))
                logger.info('The Circuit 2 has label_rank \n {}'.format(lable_rank_c2))
                logger.info('The Circuit 3 has label_rank \n {}'.format(lable_rank_c3))
                logger.info('The Circuit 4 has label_rank \n {}'.format(lable_rank_c4))
                logger.info('The Circuit 5 has label_rank \n {}'.format(lable_rank_c5))
                logger.info('The Circuit 6 has label_rank \n {}'.format(lable_rank_c6))
                logger.info('The Circuit SUM has label_rank \n {}'.format(lable_rank_sum))
                logger.info('The HIDDEN STATE has label_rank \n {}'.format(label_rank_hs))
                logger.info('####INSPIRATION####')
                logger.info('##{}-th layer ##Inspire##: The c2head1 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h1))
                logger.info('##{}-th layer ##Inspire##: The c2head2 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h2))
                logger.info('##{}-th layer ##Inspire##: The c2head3 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h3))
                logger.info('##{}-th layer ##Inspire##: The c2head4 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h4))
                logger.info('##{}-th layer ##Inspire##: The c2head5 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h5))
                logger.info('##{}-th layer ##Inspire##: The c2head6 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h6))
                logger.info('##{}-th layer ##Inspire##: The c2head7 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h7))
                logger.info('##{}-th layer ##Inspire##: The c2head8 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h8))
                logger.info('##{}-th layer ##Inspire##: The c2head9 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h9))
                logger.info('##{}-th layer ##Inspire##: The c2head10 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h10))
                logger.info('##{}-th layer ##Inspire##: The c2head11 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h11))
                logger.info('##{}-th layer ##Inspire##: The c2head12 Inspire status of source tokens is \n {}'.format(i,Inspire_c2h12))
                
                logger.info('##{}-th layer ##Inspire##: The circuit3 Inspire status of source tokens is \n {}'.format(i,Inspire_c3))
                logger.info('##{}-th layer ##Inspire##: The circuit4 Inspire status of source tokens is \n {}'.format(i,Inspire_c4))
                
                logger.info('##{}-th layer ##Inspire##: The c4head1 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h1))
                logger.info('##{}-th layer ##Inspire##: The c4head2 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h2))
                logger.info('##{}-th layer ##Inspire##: The c4head3 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h3))
                logger.info('##{}-th layer ##Inspire##: The c4head4 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h4))
                logger.info('##{}-th layer ##Inspire##: The c4head5 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h5))
                logger.info('##{}-th layer ##Inspire##: The c4head6 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h6))
                logger.info('##{}-th layer ##Inspire##: The c4head7 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h7))
                logger.info('##{}-th layer ##Inspire##: The c4head8 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h8))
                logger.info('##{}-th layer ##Inspire##: The c4head9 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h9))
                logger.info('##{}-th layer ##Inspire##: The c4head10 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h10))
                logger.info('##{}-th layer ##Inspire##: The c4head11 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h11))
                logger.info('##{}-th layer ##Inspire##: The c4head12 Inspire status of source tokens is \n {}'.format(i,Inspire_c4h12))
                
                logger.info('##{}-th layer ##Inspire##: The c4head1_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h1_all))
                logger.info('##{}-th layer ##Inspire##: The c4head2_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h2_all))
                logger.info('##{}-th layer ##Inspire##: The c4head3_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h3_all))
                logger.info('##{}-th layer ##Inspire##: The c4head4_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h4_all))
                logger.info('##{}-th layer ##Inspire##: The c4head5_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h5_all))
                logger.info('##{}-th layer ##Inspire##: The c4head6_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h6_all))
                logger.info('##{}-th layer ##Inspire##: The c4head7_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h7_all))
                logger.info('##{}-th layer ##Inspire##: The c4head8_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h8_all))
                logger.info('##{}-th layer ##Inspire##: The c4head9_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h9_all))
                logger.info('##{}-th layer ##Inspire##: The c4head10_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h10_all))
                logger.info('##{}-th layer ##Inspire##: The c4head11_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h11_all))
                logger.info('##{}-th layer ##Inspire##: The c4head12_all Inspire status of source tokens is \n {}'.format(i,Inspire_c4h12_all))
        
                logger.info('##{}-th layer ##Inspire##: The Inspire_c4compt status of source tokens is \n {}'.format(i,Inspire_c4compt))
                logger.info('##{}-th layer ##Inspire##: The circuit5 Inspire status of source tokens is \n {}'.format(i,Inspire_c5))
        
        logging.shutdown()
            
            
            
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
    
    def get_inspiration(self,input,output):
        Inspire_flag=torch.where(input>output,1,0)
        return torch.cat((Inspire_flag,input,output),dim=-1)
    
    def get_inspiration_num(self,input,output,label_ids):
        #input and output are both [1,N,E] tensors,
        #output:
            # whether the indices of label_ids in top tokens arise? 1 represents yes
            # the original rank without inpiration
            # the real rank with inpiration
            # the account of tokens that in input with higher logits than ones in output
        input=F.softmax(input,dim=-1)
        output=F.softmax(output,dim=-1)
        result=torch.sum(input>output,dim=-1)
        result=result/input.size()[-1]
        #get the label_ids 's rank
        
        sorted_indices_input=input.argsort()
        sorted_indices_output=output.argsort()
        input_rank =input.size()[-1]-sorted_indices_input.eq(label_ids).nonzero().split(1,dim=-1)[-1]
        output_rank =output.size()[-1]-sorted_indices_output.eq(label_ids).nonzero().split(1,dim=-1)[-1]
        Inspire_flag=torch.where(input_rank<output_rank,1,0)
        return torch.cat((Inspire_flag.unsqueeze(0),output_rank.unsqueeze(0),input_rank.unsqueeze(0),result.unsqueeze(-1)),dim=-1)
    
    def get_token_rank(self,representation,label_ids):      
        representation=F.softmax(representation,dim=-1)
        sorted_indices=representation.argsort()
        b=sorted_indices.eq(label_ids).nonzero().split(1,dim=-1)[-1]
        rank =representation.size()[-1]-b
        return rank
                       
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
        for i in range(predicted_indices.size()[-1]):
            ids=predicted_indices[i]
            token_list.append(self.tokenizer.decode(ids))
        return token_list         