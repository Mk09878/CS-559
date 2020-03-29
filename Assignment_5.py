"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import copy

np.random.seed(100)

#Reading the data into a numpy array
def read_idx(filename):
    with open(filename, 'rb') as file:
        temp = st.unpack('>HBB', file.read(4))
        shape = tuple(st.unpack('>I', file.read(4))[0] for d in range(temp[2]))
        data = np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)
        return data

train_data = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')
test_data = read_idx('t10k-images.idx3-ubyte')
test_labels = read_idx('t10k-labels.idx1-ubyte')

#Initializing parameters
w_init = np.random.uniform(0, 1, size = (30, 784))
w_init_bias = np.random.uniform(0, 1, size = (30, 1))
w_final = np.random.uniform(0, 1, size = (10, 30))
w_final_bias = np.random.uniform(0, 1, size = (10, 1))

""" --- Activation Functions --- """

"""
First Layer : Tanh activation function
Second Layer : Sigmoid activation function
"""

#Feedforward
def ff_init_act(u):  
    return np.tanh(u)

def ff_final_act(y): 
    return 1/ (1 + np.exp(2))

#Feedback
def fb_init_act(u):
    return (1 - np.tanh(u) ** 2)

def fb_final_act(y):
    return ff_init_act(y) * (1 - fb_init_act(y))

