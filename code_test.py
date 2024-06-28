import torch 
import torch.nn.functional as F 

a=torch.FloatTensor([1,2,3])
print(F.softmax(a))
print(F.softmax(a+a))
