import numpy as np
import cupy as cp
# import cupy as np
import pandas as pd
from collections import OrderedDict
import re
import json
from tqdm import tqdm
import pickle
dataset = pd.read_csv("train1.csv")
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
        return x.reshape(1,self.hidden_units)
with open("word_count.json",'r') as f:
    word_count = OrderedDict(json.load(f))
with open("w2v_model.pkl","rb") as f:
    model1 = pickle.load(f)
import numpy as np
import cupy as cp
# import cupy as np
import pandas as pd
from collections import OrderedDict
import re
import json
from tqdm import tqdm
import pickle
import sys
dataset = pd.read_csv("train1.csv")
np.set_printoptions(threshold=sys.maxsize)
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
        return x.reshape(1,self.hidden_units)
with open("word_count.json",'r') as f:
    word_count = OrderedDict(json.load(f))
with open("w2v_model.pkl","rb") as f:
    model1 = pickle.load(f)
class RNN:
    def __init__(self,word_count,hidden_size=300):
        self.word_count = word_count
        self.word_len = len(word_count)
        self.voc = list(word_count.keys())
        self.hidden_size = hidden_size
        self.lr = 1e-9
        self.Weight = {}
        self.Weight["Wh"] = np.random.normal(-np.sqrt(6.0/(hidden_size)), np.sqrt(6.0/(hidden_size)),(hidden_size,hidden_size))
        self.Weight["Wo"] = np.random.normal(-np.sqrt(6.0/(hidden_size)), np.sqrt(6.0/(hidden_size)),(self.word_len,hidden_size))
        self.Weight["Wi"] = np.random.normal(-np.sqrt(6.0/(hidden_size)), np.sqrt(6.0/(hidden_size)),(hidden_size,hidden_size))
        self.Weight["bh"] = np.zeros((hidden_size,1))
        self.Weight["bo"] = np.zeros((self.word_len,1))
        self.h0 = np.zeros((1,hidden_size))
    def reseter(self):
        self.onehoty = np.zeros((self.word_len,1))
        self.onehotx = np.zeros((self.word_len,1))
        self.h0 = np.zeros((1,self.hidden_size))
    def relu(self,x):
        x_shape = x.shape
        mask = x<=0
        x[mask] = 0
        y = x.reshape(x_shape)
        return y
    def relu_dev(self,x):
        x_shape = x.shape
        mask = x<=0
        x[mask] = 0
        y = x.reshape(x_shape)
        return y
    def tanh(self,x):
        z = x
        # e1 = np.exp(z)
        # e2 = np.exp(-z)
        # self.tanh_value = (e1 - e2)/(e1+e2)
        self.tanh_value = np.tanh(z)
        return self.tanh_value
    def tanh_dev(self,x):
        return np.square(1-self.tanh_value)
    def linear(self,x,w,b):
        y = np.dot(w,x.T)
        # y += b
        return y
    def softmax(self,x):
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
    def loss(self,y):    
        hadamard = np.multiply(np.log(y+1e-30),self.onehoty).T
        doter = np.dot(self.onehoty.T,hadamard.T)
        loss = -np.sum(doter)
        return loss
    def loss_dev(self,t,y):
        y = t-y
        return y
    def lookup(self,x):
        return self.word_count[re.sub(r'[^A-Za-z0-9\\]+','', x ).lower()][1]
    def forward(self,x):
        self.tanh_back = []
        self.ht = [self.h0]
        self.y = []
        self.soft_dev = []
        self.t = []
        self.xi = []
        self.ht_back = []
        self.loss_value = 0
        for idx,word in enumerate(x[:-1]):
            self.reseter()
            self.onehotx[self.lookup(word)] + 1
            embedd = model1.wv(word).get()
            xi = self.linear(embedd,self.Weight['Wi'],0)
            self.xi.append(embedd)
            self.ht_back.append(self.ht[-1])
            ht = self.linear(self.ht[-1],self.Weight['Wh'],self.Weight["bh"])
            ht = xi+ht
            self.tanh_back.append(ht)
            ht = self.tanh(ht)
            self.ht.append(ht.T)
            yt = self.relu(self.linear(ht.T,self.Weight['Wo'],self.Weight["bo"]))
            y = self.softmax(yt)
            self.soft_dev.append(y)
            self.y.append(y)
            self.onehoty[self.lookup(x[idx+1])] + 1
            self.t.append(self.onehoty)
            lossess = self.loss(y)
            self.loss_value += lossess
        return self.loss_value
    def backward(self):
        self.doter = []
        temp_wi = np.zeros((self.hidden_size,self.hidden_size))
        temp_wh = np.zeros((self.hidden_size,self.hidden_size))
        temp_wo = np.zeros((self.word_len,self.hidden_size))
        htl = np.zeros((self.hidden_size,1))
        self.soft_dev.reverse()
        self.t.reverse()
        self.ht.reverse()
        for idx in range(len(self.t)):
            x = self.loss_dev(self.soft_dev[idx],self.t[idx])
            doter = self.relu_dev(x)
            self.doter.append(doter)
            wo = np.dot(doter,self.ht[idx])
            self.Weight["bo"] -= self.lr*doter
            temp_wo += wo
        self.tanh_back.reverse()
        self.ht_back.append(np.zeros((1,self.hidden_size)))
        self.ht_back.reverse()
        self.xi.reverse()
        for idx,ht in enumerate(self.tanh_back):
            doter = self.doter[idx]
            ht = self.tanh_dev(ht)
            dx = np.dot(np.dot(doter.T,self.Weight['Wo']).T,ht.T)
            dx1 = np.dot(htl,ht.T)
            adder = dx + dx1
            wh = np.dot(adder,self.ht_back[idx].T)
            wi = np.dot(adder,self.xi[idx].T)
            htl = np.dot(htl.T,self.Weight['Wh']).T
            temp_wh += wh
            temp_wi += wi
            if np.isnan(htl).any():
                print('wit')
                exit()
            # self.Weight["bh"] -= self.lr*((dx+htl).sum(axis=1)).reshape(300,1)
        self.Weight['Wo'] -= self.lr*temp_wo
        self.Weight['Wi'] -= self.lr*temp_wi
        self.Weight['Wh'] -= self.lr*temp_wh
    def language(self,x):
        str = ' '
        str_list = []
        self.ht = [self.h0]
        for _ in range(6):
            embedd = model1.wv(x).get()
            xi = self.linear(embedd,self.Weight['Wi'],0)
            ht = self.linear(self.ht[-1],self.Weight['Wh'],self.Weight["bh"])
            ht = xi+ht
            ht = self.tanh(ht)
            self.ht.append(ht.T)
            yt = self.relu(self.linear(ht.T,self.Weight['Wo'],self.Weight["bo"]))
            y = self.softmax(yt)
            x = np.argmax(y)
            x = self.voc[x]
            str_list.append(x)
        str = str.join(str_list)
        print(str)
        return str
model = RNN(word_count)
for _ in tqdm(dataset.index[:2000]):
    sentence = dataset.iloc[_]['target'].split(' ')
    y = model.forward(sentence)
    model.backward()
model.language("dog")
def exit_op():
    with open('./RNN_model.pkl','wb') as f:
        pickle.dump(model, f)
            
            
        
        
        
        
        
        