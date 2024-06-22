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
                       