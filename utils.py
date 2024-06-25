import pickle, json, decimal, math,os
import numpy as np
from typing import Union
from dataset.ioi_dataset import IOIDataset
import matplotlib.pyplot as plt
from matplotlib import pyplot


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js


def set_default_to_empty_string(v, default_v, activate_flag):
    if ((default_v is not None and v == default_v) or (
            default_v is None and v is None)) and (activate_flag):
        return ""
    else:
        return f'_{v}'
    
def set_gpu(gpu_id: Union[str, int]):
    if isinstance(gpu_id, int):
        gpu_id = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    



def get_model_layer_num(model = None):
    num_layer = None
    if model is not None:
        if hasattr(model.config, 'num_hidden_layers'):
            num_layer = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layers'):
            num_layer = model.config.n_layers
        elif hasattr(model.config, 'n_layer'):
            num_layer = model.config.n_layer
        else:
            pass
    if num_layer is None:
        raise ValueError(f'cannot get num_layer from model: {model} or model_name: {model_name}')
    return num_layer



def get_datasets():
    """from unity"""
    batch_size = 500
    orig = "When John and Mary went to the store, John gave a bottle of milk to Mary"
    new = "When Alice and Bob went to the store, Charlie gave a bottle of milk to Mary"
    prompts_orig = [
        {"S": "John", "IO": "Mary", "TEMPLATE_IDX": -42, "text": orig}
    ]  # TODO make ET dataset construction not need TEMPLATE_IDX
    prompts_new = [{"S": "Alice", "IO": "Bob", "TEMPLATE_IDX": -42, "text": new}]
    prompts_new[0]["text"] = new
    dataset_orig = IOIDataset(
        N=batch_size, prompt_type="mixed"
    )  # TODO make ET dataset construction not need prompt_type
    dataset_new = IOIDataset(
        N=batch_size,
        prompt_type="mixed",
        manual_word_idx=dataset_orig.word_idx,
    )
    return dataset_new, dataset_orig


def generate_figure(initial_token_all,emerge_token_all,predicted_token_all):
    plt.style.use('seaborn-v0_8-whitegrid')
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 32,
    }
    initial_0=[]
    initial_1=[]
    initial_2=[]
    initial_3=[]
    initial_4=[]
    initial_5=[]
    initial_6=[]
    initial_7=[]
    initial_8=[]
    initial_9=[]
    
    emerge_0=[]
    emerge_1=[]
    emerge_2=[]
    emerge_3=[]
    emerge_4=[]
    emerge_5=[]
    emerge_6=[]
    emerge_7=[]
    emerge_8=[]
    emerge_9=[]
    emerge_10=[]
    emerge_11=[]
    emerge_12=[]
    emerge_13=[]
    emerge_14=[]
    emerge_15=[]
    emerge_16=[]
    emerge_17=[]
    emerge_18=[]
    emerge_19=[]
    emerge_20=[]
    emerge_21=[]
    emerge_22=[]
    emerge_23=[]
    emerge_24=[]
    emerge_25=[]
    emerge_26=[]
    emerge_27=[]
    emerge_28=[]
    emerge_29=[]
    
    predicted_0=[]
    predicted_1=[]
    predicted_2=[]
    predicted_3=[]
    predicted_4=[]
    predicted_5=[]
    predicted_6=[]
    predicted_7=[]
    predicted_8=[]
    predicted_9=[]
    for i in range(len(initial_token_all)):
        initial_0.append(initial_token_all[i][0])
        initial_1.append(initial_token_all[i][1])
        initial_2.append(initial_token_all[i][2])
        initial_3.append(initial_token_all[i][3])
        initial_4.append(initial_token_all[i][4])
        initial_5.append(initial_token_all[i][5])
        initial_6.append(initial_token_all[i][6])
        initial_7.append(initial_token_all[i][7])
        initial_8.append(initial_token_all[i][8])
        initial_9.append(initial_token_all[i][9])
        
        predicted_0.append(predicted_token_all[i][0])
        predicted_1.append(predicted_token_all[i][1])
        predicted_2.append(predicted_token_all[i][2])
        predicted_3.append(predicted_token_all[i][3])
        predicted_4.append(predicted_token_all[i][4])
        predicted_5.append(predicted_token_all[i][5])
        predicted_6.append(predicted_token_all[i][6])
        predicted_7.append(predicted_token_all[i][7])
        predicted_8.append(predicted_token_all[i][8])
        predicted_9.append(predicted_token_all[i][9])
        
        if len(emerge_token_all[i])>0: emerge_0.append(emerge_token_all[i][0])
        if len(emerge_token_all[i])>1: emerge_1.append(emerge_token_all[i][1])
        if len(emerge_token_all[i])>2: emerge_2.append(emerge_token_all[i][2])
        if len(emerge_token_all[i])>3: emerge_3.append(emerge_token_all[i][3])
        if len(emerge_token_all[i])>4: emerge_4.append(emerge_token_all[i][4])
        if len(emerge_token_all[i])>5: emerge_5.append(emerge_token_all[i][5])
        if len(emerge_token_all[i])>6: emerge_6.append(emerge_token_all[i][6])
        if len(emerge_token_all[i])>7: emerge_7.append(emerge_token_all[i][7])
        if len(emerge_token_all[i])>8: emerge_8.append(emerge_token_all[i][8])
        if len(emerge_token_all[i])>9: emerge_9.append(emerge_token_all[i][9])
        if len(emerge_token_all[i])>10: emerge_10.append(emerge_token_all[i][10])
        if len(emerge_token_all[i])>11: emerge_11.append(emerge_token_all[i][11])
        if len(emerge_token_all[i])>12: emerge_12.append(emerge_token_all[i][12])
        if len(emerge_token_all[i])>13: emerge_13.append(emerge_token_all[i][13])
        if len(emerge_token_all[i])>14: emerge_14.append(emerge_token_all[i][14])
        if len(emerge_token_all[i])>15: emerge_15.append(emerge_token_all[i][15])
        if len(emerge_token_all[i])>16: emerge_16.append(emerge_token_all[i][16])
        if len(emerge_token_all[i])>17: emerge_17.append(emerge_token_all[i][17])
        if len(emerge_token_all[i])>18: emerge_18.append(emerge_token_all[i][18])
        if len(emerge_token_all[i])>19: emerge_19.append(emerge_token_all[i][19])
        if len(emerge_token_all[i])>20: emerge_20.append(emerge_token_all[i][20])
        if len(emerge_token_all[i])>21: emerge_21.append(emerge_token_all[i][21])
        if len(emerge_token_all[i])>22: emerge_22.append(emerge_token_all[i][22])
        if len(emerge_token_all[i])>23: emerge_23.append(emerge_token_all[i][23])
        if len(emerge_token_all[i])>24: emerge_24.append(emerge_token_all[i][24])
        if len(emerge_token_all[i])>25: emerge_25.append(emerge_token_all[i][25])
        if len(emerge_token_all[i])>26: emerge_26.append(emerge_token_all[i][26])
        if len(emerge_token_all[i])>27: emerge_27.append(emerge_token_all[i][27])
        if len(emerge_token_all[i])>28: emerge_28.append(emerge_token_all[i][28])
        if len(emerge_token_all[i])>29: emerge_29.append(emerge_token_all[i][29])
    
    
    
    draw_line("initial",1,initial_0)
    #draw_line("initial",1,initial_1)
    # draw_line("initial",1,initial_2)
    # draw_line("initial",1,initial_3)
    # draw_line("initial",1,initial_4)
    # draw_line("initial",1,initial_5)
    # draw_line("initial",1,initial_6)
    # draw_line("initial",1,initial_7)
    # draw_line("initial",1,initial_8)
    # draw_line("initial",1,initial_9)
    
    draw_line("predicted",2,predicted_0)
    draw_line("predicted",2,predicted_1)
    draw_line("predicted",2,predicted_2)
    draw_line("predicted",2,predicted_3)
    draw_line("predicted",2,predicted_4)
    # draw_line("predicted",2,predicted_5)
    # draw_line("predicted",2,predicted_6)
    # draw_line("predicted",2,predicted_7)
    # draw_line("predicted",2,predicted_8)
    # draw_line("predicted",2,predicted_9)
    
    # draw_line("emerge",3,emerge_0)
    # draw_line("emerge",3,emerge_1)
    # draw_line("emerge",3,emerge_2)
    # draw_line("emerge",3,emerge_3)
    # draw_line("emerge",3,emerge_4)
    # draw_line("emerge",3,emerge_5)
    # draw_line("emerge",3,emerge_6)
    # draw_line("emerge",3,emerge_7)
    # draw_line("emerge",3,emerge_8)
    # draw_line("emerge",3,emerge_9)
    # draw_line("emerge",3,emerge_10)
    # draw_line("emerge",3,emerge_11)
    # draw_line("emerge",3,emerge_12)
    # draw_line("emerge",3,emerge_13)
    # draw_line("emerge",3,emerge_14)
    # draw_line("emerge",3,emerge_15)
    # draw_line("emerge",3,emerge_16)
    # draw_line("emerge",3,emerge_17)
    draw_line("emerge",3,emerge_18)
    draw_line("emerge",3,emerge_19)
    draw_line("emerge",3,emerge_20)
    draw_line("emerge",3,emerge_21)
    draw_line("emerge",3,emerge_22)
    draw_line("emerge",3,emerge_23)
    draw_line("emerge",3,emerge_24)
    # draw_line("emerge",3,emerge_25)
    # draw_line("emerge",3,emerge_26)
    # draw_line("emerge",3,emerge_27)
    # draw_line("emerge",3,emerge_28)
    # draw_line("emerge",3,emerge_29)
    


    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('layer',fontsize=50)
    plt.ylabel('logits',fontsize=50)
    plt.legend(loc='upper left',prop=font1)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    # plt.xlabel('Token',fontsize=32)
    # plt.ylabel('E(l)',fontsize=32)
    # plt.legend(loc='upper left',prop=font1)


    plt.savefig('paper_figure/top_token_circuit1.jpg')
    # plt.savefig('paper_figure/context_GPTXL_2.jpg')
    plt.show()



def draw_line(name_of_alg,color_index,datas):
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = pyplot.get_cmap('Set1')
    
    
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    
    ##################
    x_values = range(0, len(datas[0]), 1)
    plt.xticks(x_values)
    plt.plot(x_values, avg, color=color, linewidth=3)
    plt.fill_between(x_values, r1, r2, color=color, alpha=0.2)
    ##################
    # plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3)
    # plt.fill_between(iters, r1, r2, color=color, alpha=0.2)