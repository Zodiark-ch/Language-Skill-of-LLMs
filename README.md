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
        We design a class 'assert_circuits_equal_output' to show the decouplement of forward.  
        All consists of 6 circuits, 
        c1 represents the input self. 
        c2 represents the attention-only circuits, showing the contribution of unique attention. 
        c3 represents the MLP-only circuits, showing the contribution of unique MLP. 
        c4 represents the attention+MLP circuits, showing the contribution of path through attention followed by mlp. 
        c5 represents the synergistic circuits, showing the synergistic contribution of bias(Wmlp1) to residual and attention to mlp. 
            It means when residual stream was decoupled into several circuits, the loss distribution caused by activation. 
        c6 represents the translation circuits, it add a bias to vocabulary distribution. 

        Their sum is equal to the original output representation. 

    7, How does each circuit perform in vocabulary space?
        Some has clear senmantic, showed in class show_vocabulary_circuit in details. 

    8, A dataset-wise analysis for attention circuit (circuit 1 and 2):
        
