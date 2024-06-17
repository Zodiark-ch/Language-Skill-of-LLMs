# XAIofLLMs 

The first step: finish and observe the vocabulary mapping of output in each layer 
    1, How to construct a vocabulary len from logits to token? 
        _,predicted_indices=torch.topk(outputs.logits[0][-1],10)
        print('max probability token_ids are:', predicted_indices)
        print('max probability tokens are:', tokenizer.decode(predicted_indices))

    2, How to map the representation to logits? 
        lm_logits = self.lm_head(hidden_states) (modeling_gpt2.py) from [B, N, D] to [B, N, E] (in gpt2, D is 768 and E is 50257) 

    3, Is there different between FFN(original forward in one transformer layer, not only the mlp) and matrix producting? 
        Yes, we design a class 'assert_FFNandproduction_gpt2xl' to show how similar or even the same between these two operations 

    4, How to get the vocabulary distribution of output representation in each layer? 
        we design a class 'show_each_layer_vocb' to show the vocab space of output representation, breifly, let the hidden_state to be layer_normed and unembedded. and we found that ln_f is necessary. 

    5, Are logits of stream equal to the original output logits in each layer? 
        Yes, we design a class 'assert_attentionmlp_equal_output' to show the vocab space of stream logits, which corresponds the logits from original representation from FFN. 

    6, Are all circuits' addtion equal to the original output representation?  
