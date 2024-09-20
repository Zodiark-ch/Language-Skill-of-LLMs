import networkx as nx
import matplotlib.pyplot as plt
import json

with open('json_logs/token_by_token/gpt2xl/'+'language_skill.json','r') as f: 
    input_text=json.load(f)

plt.figure(figsize=(120, 160))
G = nx.DiGraph()

node_list=[]
edge_list=[]
layer0=[]
layer1=[]
layer2=[]
layer3=[]
layer4=[]
layer5=[]
layer6=[]
layer7=[]
layer8=[]
layer9=[]
layer10=[]
layer11=[]
for chain in range(len(input_text)):
    chain_weight=input_text[chain]
    node_num=len(chain_weight)-1

    for node_idx in range(2):
        if chain_weight[node_idx] not in node_list:
            G.add_node('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==0:
                layer0.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==1:
                layer1.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==2:
                layer2.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==3:
                layer3.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==4:
                layer4.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==5:
                layer5.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==6:
                layer6.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==7:
                layer7.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==8:
                layer8.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==9:
                layer9.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==10:
                layer10.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
            if chain_weight[node_idx]//29==11:
                layer11.append('[{}, {}]'.format(chain_weight[node_idx]//29, chain_weight[node_idx]%29))
    edge=['[{}, {}]'.format(chain_weight[0]//29, chain_weight[0]%29), '[{}, {}]'.format(chain_weight[1]//29, chain_weight[1]%29)]
    if edge not in edge_list:
        G.add_edge('[{}, {}]'.format(chain_weight[0]//29, chain_weight[0]%29), '[{}, {}]'.format(chain_weight[1]//29, chain_weight[1]%29), weight=round(chain_weight[2],2))
        edge_list.append(edge)

shell_layout=[]
if len(layer11)>0:
    shell_layout.append(layer11)
if len(layer10)>0:
    shell_layout.append(layer10)
if len(layer9)>0:
    shell_layout.append(layer9)
if len(layer8)>0:
    shell_layout.append(layer8)
if len(layer7)>0:
    shell_layout.append(layer7)
if len(layer6)>0:
    shell_layout.append(layer6)
if len(layer5)>0:
    shell_layout.append(layer5)
if len(layer4)>0:
    shell_layout.append(layer4)
if len(layer3)>0:
    shell_layout.append(layer3)
if len(layer2)>0:
    shell_layout.append(layer2)
if len(layer1)>0:
    shell_layout.append(layer1)
if len(layer0)>0:
    shell_layout.append(layer0)

pos = nx.shell_layout(G,shell_layout)
#pos = nx.spectral_layout(G)

    
trans_blue = (0.16, 0.47, 0.71, 0.5)
blue = (0.16, 0.47, 0.71, 1)
red=(0.78,0.14,0.14)
orange=(0.97,0.67,0.55)
#nx.draw(G, pos, with_labels=False, arrows=True)
nx.draw_networkx_nodes(G, pos, node_color=trans_blue, node_size=4000)
nx.draw_networkx_edges(G, pos, edge_color=blue)
nx.draw_networkx_labels(G, pos, font_size=40, font_color=red)

edge_labels = nx.get_edge_attributes(G, 'weight')


nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=16, font_color=orange)


plt.show()
plt.savefig('paper_figure/icl_qa_skill_graph.jpg')