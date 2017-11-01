# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:08:04 2017

@author: esppk

"""
import pandas as pd 
import sys
input = str(sys.argv[1])
output = str(sys.argv[2])

X = pd.read_csv(input)
X_train = X[[0,1]].values
y_train = X[[2]].values
#%%
import numpy as np
#%%
def sign_(x):
    if x == 0:
        return int(-1)
    else:
        return int(np.sign(x))
        
    
def perceptron(X_train, y_train):
    output_weight = []
    n = X_train.shape[0]
    w = np.zeros(X_train.shape[1]+1)
    ones =  np.ones([n, 3])
    ones[:,:-1] = X_train
    X_train = ones
    while(True):
#    for ii in range(10000):
        output_weight.append(w)
        check_list = []
        for i in range(n):
            
            x_i = X_train[i]
            y_i = y_train[i]
            
            y_hat = sign_(np.dot(x_i, w))
            
            check = bool(y_i == y_hat)
            
            if check == False:
               
                check_list.append(i)
        
        if check_list != []:
            idx = np.random.choice(check_list)
            x_i = X_train[idx]
            y_i = y_train[idx]
            ita = 1
            w = w + ita*y_i*x_i
        else:
            return output_weight
            
#%%
#train
output_weight = perceptron(X_train, y_train)            
            
            
#%%
#prediction:
w = output_weight[-1]
pred = []
for i in range(X_train.shape[0]):
    x_i = X_train[i]
    p = sign_(np.dot(x_i,w[:-1]) + w[-1] )
    pred.append(p)          
            
            
pred = np.array(pred)      

      
            
np.savetxt(output, output_weight, delimiter=",") # write output to file
            
            
            
            
            
            
            
            
            
            
            