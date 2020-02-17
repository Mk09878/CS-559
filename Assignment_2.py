"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import time
import struct as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

np.random.seed(100)

#Reading the data into a numpy array
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

train_data = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')
test_data = read_idx('t10k-images.idx3-ubyte')
test_labels = read_idx('t10k-labels.idx1-ubyte')

#Initializing the parameters
n = 60000
learning_rate = 0.5
e = 0.15
W = np.random.uniform(-1, 1, size = (10, 784))
epoch = 0
errors_epoch = [0] * n

d = {}
temp_enc = np.zeros((10,1))
for x in range(10):
    temp_enc = np.zeros((10,1))
    temp_enc[x] = 1
    d[x] = temp_enc

#The step activation function    
def step_act(y):
    for x in range(len(y)):
        if(y[x] >= 0):
            y[x] = 1
        else:
            y[x] = 0
    return y

#Training the neural network (Updating the weights)              
while(True):
    for i in range(n):
        x = train_data[i].reshape(784,1)
        temp = np.dot(W, x)
        if(temp.argmax() != train_labels[i]):
            errors_epoch[epoch] += 1
            
    epoch += 1
    
    for i in range(n):
        x = train_data[i].reshape(784,1)
        test = step_act(np.dot(W, x))
        test1 = d[train_labels[i]]
        test2 = (d[train_labels[i]] - step_act(np.dot(W, x)))
        test3 = learning_rate * (d[train_labels[i]] - step_act(np.dot(W, x))) * x.T
        W = W + learning_rate * (d[train_labels[i]] - step_act(np.dot(W, x))) * x.T
        
    print(errors_epoch[epoch - 1]/n)
    if(errors_epoch[epoch - 1]/n <= e):
        break

#Testing the neural network
no_of_errors = 0
for i in range(len(test_data)):
    x = test_data[i].reshape(784,1)
    temp = np.dot(W, x)
    if(temp.argmax() != test_labels[i]):
        no_of_errors += 1
       
    
    

