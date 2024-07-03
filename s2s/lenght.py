# import pandas as pd
import torch
torch.set_printoptions(profile="full")
# dataset = pd.read_csv("s2s/train1.csv")
# max_l = 0
# for _ in dataset.index:
#         string = dataset.iloc[_]['target'].split(" ")
#         if max_l <len(string):
#             max_l = len(string)
# print(max_l)
# def Pos_encode(dim,lenght,n=10000):
#         pe = torch.zeros(lenght,dim)
#         # for pos in range(lenght):
#         #     for i in range(0,int(dim/2),1):
#         #         pow_n = pow(n,(2*i)/dim)
#         #         pe[pos,2*i] = torch.sin(torch.divide(pos,pow_n))
#         #         pe[pos,2*i+1] = torch.cos(torch.divide(pos,pow_n))
#         print(pe[:,0])
#         # with open("s2s/a.txt","w") as f:
#         #     f.write(str(pe))
# Pos_encode(300,20)
x = torch.zeros((10,10))
print(x)
print(torch.std(x))