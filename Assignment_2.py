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

W_old = np.random.uniform(-1, 1, size = (10, 784))

#Onehotencoding the labels
d = {}
temp_enc = np.zeros((10,1))
for x in range(10):
    temp_enc = np.zeros((10,1))
    temp_enc[x] = 1
    d[x] = temp_enc

#Implementing the step activation function    
def step_act(y):
    for x in range(len(y)):
        if(y[x] >= 0):
            y[x] = 1
        else:
            y[x] = 0
    return y

#Training the neural network (Updating the weights)
def training(n, learning_rate, e):
    W = copy.deepcopy(W_old)
    epoch = 0
    errors_epoch = [0] * n          
    while(True):
        
        for i in range(n):
            x = train_data[i].reshape(784,1)
            temp = np.dot(W, x)
            if(temp.argmax() != train_labels[i]):
                errors_epoch[epoch] += 1
                
        epoch += 1
        
        for i in range(n):
            x = train_data[i].reshape(784,1)
            W = W + learning_rate * (d[train_labels[i]] - step_act(np.dot(W, x))) * x.T
            
        if(errors_epoch[epoch - 1]/n <= e or epoch > 70):
            break
    print("----------For n = {}, learning_rate = {}, e = {}----------\n".format(n, learning_rate, e))
    print("Number of epochs:",epoch)
    epoch_arr = range(epoch)
    plt.title("Epoch Number vs Number of Misclassifications")
    plt.plot(epoch_arr, errors_epoch[0:epoch], c = 'black')
    plt.xlabel("Epoch Number")
    plt.ylabel("Number of Misclassifications")
    plt.show()

    return W
    
#Testing the neural network
def test(W):    
    no_of_errors = 0
    for i in range(len(test_data)):
        x = test_data[i].reshape(784,1)
        temp = np.dot(W, x)
        if(temp.argmax() != test_labels[i]):
            no_of_errors += 1
    print("Percentage of testing error:", no_of_errors/len(test_data) * 100)
    print("")

W = training(100, 0.5, 0)
test(W)

W = training(50, 1, 0)
test(W)    

W = training(1000, 1, 0)
test(W) 

"""training(60000, 1, 0)

W_old = np.random.uniform(-1, 1, size = (10, 784))
W = training(60000, 1, 0.14)
test(W) 

W_old = np.random.uniform(-1, 1, size = (10, 784))
W = training(60000, 1, 0.14)
test(W)

W_old = np.random.uniform(-1, 1, size = (10, 784))
W = training(60000, 1, 0.14)
test(W)
"""

