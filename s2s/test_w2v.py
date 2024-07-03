import numpy as np
import cupy as cp
import json
import pickle
import pandas as pd
from collections import OrderedDict
import re
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=sys.maxsize)
with open("word_count.json",'r') as f:
    w_json = json.load(f)
    word_count = OrderedDict(w_json)
class Skip_Gram:
    def __init__(self,word_count,window = 4,hidden_units = 300):
        self.window = window
        self.learningrate = 1e-12
        len_word = len(word_count)
        self.word_count = word_count
        self.hidden_units = hidden_units
        self.len_word = len_word
        self.reseter()
        self.Weight = {}
        self.Weight["W"+str(1)] = np.random.normal(-1., 1.,(len_word,hidden_units)) 
        self.Weight["W"+str(2)] = np.random.normal(-1., 1.,(hidden_units,len_word))
        # self.Weight["W"+str(1)] = np.random.randn(len_word,hidden_units)/2.5+1
        # self.Weight["W"+str(2)] = np.random.randn(len_word,hidden_units)
        # self.Weight["W"+str(2)] = np.random.randn(hidden_units,len_word)/2.5+1
    def reseter(self):
        self.onehoty = cp.zeros((self.len_word,1))
        self.onehotx = cp.zeros((self.len_word,1))
    def softmax(self, x):
        max_val = np.max(x)
        x = x - max_val
        # if np.isnan(x).any():
        #     print("NAN")
        #     print("\n",max_val,"\n")
        #     exit_op()
        #     exit()
        x = np.exp(x)
        self.soft = x/np.sum(x)
        return self.soft
    def linear_ohot(self,x,x1):
        self.iidx = self.word_count[re.sub(r'[^A-Za-z0-9\\]+','', x ).lower()][1]
        self.onehotx[self.iidx] +=1
        y = np.dot(self.onehotx.T,x1)
        return y
    def linear(self,x,x1):
        self.x_out = x
        y = np.dot(x,x1)
        return y
    def loss(self,y):    
        hadamard = np.multiply(np.log(y+1e-25),self.onehoty.T).T
        doter = np.dot(self.onehoty.T,hadamard)
        loss = -np.sum(doter)
        return loss
    def forward(self,x,y):
        x = self.linear_ohot(x,self.Weight['W1'])
        for idx,target in enumerate(y):
            self.onehoty[self.word_count[re.sub(r'[^A-Za-z0-9\\]+','', target ).lower()][1]] += 1 
        x = self.softmax(self.linear(x,self.Weight['W2']))
        y = self.loss(x)
        return y
    def backward(self,x):
        x =  self.soft - self.onehoty.T
        b = np.multiply(self.onehoty.T,x)
        dwo = np.dot(self.x_out.T,b)
        self.Weight['W2'] -= self.learningrate*dwo
        y = np.dot(self.Weight['W2'],x.T)
        dwi = np.multiply(self.onehotx.reshape(self.onehotx.shape[0],1),y.T)
        # if np.isnan(dwi).any():
        #     print(self.onehotx)
        self.Weight['W1'] -= self.learningrate*dwi
        self.reseter()
    def predict(self,x):
        x = self.linear_ohot(x,self.Weight['W1'])
        x = self.softmax(self.linear(x,self.Weight['W'+str(2)]))
        self.reseter()
        return x
    def ss(self,x):
        text = x
        x = self.linear_ohot(x,self.Weight['W1'])
        x_values = x.reshape(300)
        x = self.linear(x,self.Weight['W'+str(2)])
        y_values = x
        self.reseter()
        word_wieght.append(x_values.get())
    def wv(self,x):
        self.reseter()
        x = self.linear_ohot(x,self.Weight['W1'])
        return x
with open("w2v_model.pkl","rb") as f:
    model = pickle.load(f)
with open("w2v_weight.pkl","rb") as f:
    weight = pickle.load(f)
    # print((weight['W1']-model.Weight['W1']).sum())
# y_idx = model.predict("open").argmax()
word_lists = list(word_count.keys())
pca = PCA(n_components=3)
words = []
word_wieght = []
print(word_lists[int(model.predict("white").argmax())])
# plt.figure(figsize=(10, 10))
fig = plt.figure(figsize=(10, 10))  # Optional: adjusting figure size
ax = fig.add_subplot(111, projection='3d')
# for _ in range(1000):
for word in ['dog','dogs','over','above']:
    # word = np.random.choice(word_lists)
    words.append(word)
    model.ss(word)
twodim = pca.fit_transform(np.array(word_wieght))
# plt.scatter(twodim[:,0], twodim[:,1], color='blue', alpha=0.4)  # 'alpha' sets transparency
ax.scatter(twodim[:,0], twodim[:,1], twodim[:,2], color=np.random.rand(3,), s=100,alpha=0.4)
for word, (x,y,z) in zip(words, twodim):
    # plt.text(x+0.05, y+0.05, f'{word}', fontsize=3, color='black', ha='center', va='center',alpha=0.4)
    ax.text(x+0.05, y+0.05, z+0.05, f'{word}', fontsize=3, color='black', ha='left', va='top',alpha=0.7)
# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Array')

# Display the plot
plt.grid(True)  # Optional: adding a grid
plt.savefig('scatter_plot.png', dpi=1000)