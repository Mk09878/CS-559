"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import struct as st
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import scale

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

train_data = train_data[0:100]
train_labels = train_labels[0:100]

#Onehotencoding the labels
d = {}
temp_enc = np.zeros((10,1))
for x in range(10):
    temp_enc = np.zeros((10,1))
    temp_enc[x] = 1
    d[x] = temp_enc
    
#Initializing parameters
w_init = np.random.normal(0, 1, size = (30, 784))
w_init_bias = np.random.normal(1, 1, size = (30, 1))
w_final = np.random.normal(0, 1, size = (10, 30))
w_final_bias = np.random.normal(1, 1, size = (10, 1))
learning_rate = 0.1
mean = np.mean(train_data.flatten())
std_dev = np.std(train_data.flatten())
induced_local_field_init = [0] * 30                                                 #Induced local field from initial layer
output_init = [0] * 30                                                 #Output from initial layer
induced_local_field_final = [0] * 10                                                 #Induced local field from final layer
output_final = [0] * 10
coded_label = [0] * 10
""" --- Activation Functions --- """

"""
First Layer : Tanh activation function
Second Layer : Sigmoid activation function
"""

#Feedforward
def ff_init_act(u):  
    return np.tanh(u)

def ff_final_act(y): 
    return 1/ (1 + np.exp(-y))

#Feedback
def fb_init_act(u):
    return (1 - np.tanh(u) ** 2)

def fb_final_act(y):
    return ff_init_act(y) * (1 - fb_init_act(y))


""" --- Feedforward Training --- """
def feed_forward():
    
    global output_final
    global coded_label
    
    for j in range(30):
        induced_local_field_init[j] = np.matmul(w_init[j], train_data_temp) + w_init_bias[j]
        output_init[j] = ff_init_act(induced_local_field_init[j])
    
    for j in range(10):
        induced_local_field_final[j] = np.matmul(w_final[j], output_init) + w_final_bias[j]
        output_final[j] = ff_final_act(induced_local_field_final[j])
        
    output_final = d[np.argmax(output_final)]
    
    coded_label = d[train_labels[i]]

        
""" --- Backpropagation --- """
def feed_back():
    
    
    
    pass
    
    
    
while(1):
    for i in range(100):
        train_data_temp = train_data[i].flatten()
        train_data_temp = (train_data_temp - mean) / std_dev
        feed_forward()
        break
    break
    