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

#train_data = train_data[0:100]
#train_labels = train_labels[0:100]
#test_data = test_data[0:100]
#test_labels = test_labels[0:100]

#Onehotencoding the labels
dictionary = {}
temp_enc = np.full((10,1), 0.1)
for x in range(10):
    temp_enc = np.full((10,1), 0.1)
    temp_enc[x] = 0.9
    dictionary[x] = temp_enc
    
#Initializing parameters
input_neurons = 784
hidden_neurons = 50
output_neurons = 10
w_init = np.random.normal(0, 1, size = (hidden_neurons, input_neurons))
w_init_bias = np.random.normal(1, 1, size = (hidden_neurons, 1))
w_final = np.random.normal(0, 1, size = (output_neurons, hidden_neurons))
w_final_bias = np.random.normal(1, 1, size = (output_neurons, 1))
grad_init = np.zeros((hidden_neurons, input_neurons))
grad_init_bias = np.zeros((hidden_neurons, 1))
grad_final = np.zeros((output_neurons, hidden_neurons))
learning_rate_layer1 = 15
learning_rate_layer2 = 5
learning_rate = 0.3
mean = np.mean(train_data.flatten())
std_dev = np.std(train_data.flatten())
induced_local_field_init = np.zeros((hidden_neurons, 1))                                                 
output_init = np.zeros((hidden_neurons, 1))                                                 
induced_local_field_final = np.zeros((output_neurons, 1))                                                 
output_final = np.zeros((output_neurons, 1))
coded_label = np.zeros((output_neurons, 1))
count = 0
correct_train_pred = 0
correct_test_pred = 0
epochs = 0
accuracy_train_arr = []
error_train_arr = []
mse_train = 0
mse_train_arr = []
accuracy_test_arr = []
error_test_arr = []
mse_test = 0
mse_test_arr = []
""" --- Activation Functions --- """

"""
First Layer : Tanh activation function
Second Layer : Sigmoid activation function
"""

#Feedforward
def ff_init_act(u):  
    if u < 0:
        return 0
    else:
        return u
def ff_final_act(y_ff): 
    return 1/ (1 + np.exp(-y_ff))

#Feedback
def fb_init_act(u):
    temp = copy.deepcopy(u)
    for a in range(len(temp)):
        if (temp[a] < 0):
            temp[a] = 0
        else:
            temp[a] = 1
    return temp

def fb_final_act(y_fb):
    return 1/ (1 + np.exp(-y_fb)) * (1 - 1/ (1 + np.exp(-y_fb)))


while(1):
        
    for i in range(len(train_data)):
        
        train_data_temp = train_data[i].flatten().reshape((784, 1))
        train_data_temp = (train_data_temp - mean) / std_dev
        
        """ --- Feedforward Training --- """
        
        for j in range(hidden_neurons):
            induced_local_field_init[j] = np.matmul(w_init[j], train_data_temp) + w_init_bias[j]
            output_init[j] = ff_init_act(induced_local_field_init[j])
    
        for j in range(output_neurons):
            induced_local_field_final[j] = np.matmul(w_final[j], output_init) + w_final_bias[j]
            output_final[j] = ff_final_act(induced_local_field_final[j])
        
        temp_final = copy.deepcopy(dictionary[np.argmax(output_final)][:])
        
        coded_label = copy.deepcopy(dictionary[train_labels[i]][:])
        mse_train += np.linalg.norm(coded_label - output_final) ** 2
        temp_mse_train = np.linalg.norm(coded_label - temp_final) ** 2
        
        if(temp_mse_train == 0):
            correct_train_pred += 1
        
        """ --- Backpropagation --- """
        
        error = coded_label - output_final
    
        #Calculating the gradients
        grad_final_bias = -1 * error * fb_final_act(induced_local_field_final)
        grad_final = (-1 * np.dot(output_init ,(error * fb_final_act(induced_local_field_final[:])).T) ).T
        grad_init_bias = -1 * fb_init_act(induced_local_field_init[:]) * np.dot(w_final.T, (error * fb_final_act(induced_local_field_final)))
        grad_init = (-1 * np.dot(train_data_temp, (fb_init_act(induced_local_field_init) * np.dot(w_final.T, (error * fb_final_act(induced_local_field_final)))).T)).T
        
        #Updating the weights
        #temp = w_init - learning_rate * (grad_init/len(train_data))
        w_init = w_init - learning_rate * (grad_init)#/len(train_data))
        w_init_bias = w_init_bias - learning_rate * (grad_init_bias)#/len(train_data))
        w_final = w_final - (learning_rate/1.5) * (grad_final)#/len(train_data))
        w_final_bias = w_final_bias - (learning_rate/1.5) * (grad_final_bias)#/len(train_data))
    
    
    mse_train = mse_train / len(train_data)
    mse_train_arr.append(mse_train)
    accuracy_train = correct_train_pred / len(train_data)
    error_train_arr.append((1 - accuracy_train) * 100)
    accuracy_train_arr.append(accuracy_train)
    print("Training accuracy = {}, learning rate = {} and number of epochs = {}".format(accuracy_train, learning_rate, epochs))
    
    
    """ --- Testing --- """
    #correct_pred_test = 0
    mean_test = np.mean(test_data.flatten())
    std_dev_test = np.std(test_data.flatten())
    for i in range(len(test_data)):
        test_data_temp = test_data[i].flatten().reshape((784, 1))
        test_data_temp = (test_data_temp - mean_test) / std_dev_test
            
        """ --- Feedforward Training --- """
            
        for j in range(hidden_neurons):
            induced_local_field_init[j] = np.matmul(w_init[j], test_data_temp) + w_init_bias[j]
            output_init[j] = ff_init_act(induced_local_field_init[j])
        
        for j in range(output_neurons):
            induced_local_field_final[j] = np.matmul(w_final[j], output_init) + w_final_bias[j]
            output_final[j] = ff_final_act(induced_local_field_final[j])
            
        temp_final = copy.deepcopy(dictionary[np.argmax(output_final)][:])
            
        coded_label = copy.deepcopy(dictionary[test_labels[i]][:])
            
        temp_mse_test = np.linalg.norm(coded_label - temp_final) ** 2
        mse_test = np.linalg.norm(coded_label - output_final) ** 2
        if(temp_mse_test == 0):
            correct_test_pred += 1
    
    mse_test = mse_test / len(train_data)
    mse_test_arr.append(mse_test)
    accuracy_test = correct_test_pred / len(test_data)
    error_test_arr.append((1 - accuracy_test) * 100)
    accuracy_test_arr.append(accuracy_test)
    print("Test accuracy = {}, learning rate = {} and number of epochs = {}".format(accuracy_test, learning_rate, epochs))
    
    correct_test_pred = 0
    mse_test = 0
    mse_train = 0
    correct_train_pred = 0
    
    if(mse_train_arr[epochs] > mse_train_arr[epochs - 1]):
        #learning_rate_layer1 = learning_rate_layer1 * 0.9
        #learning_rate_layer2 = learning_rate_layer2 * 0.97
        learning_rate = learning_rate * 0.9
    
    epochs += 1
    
    
        
    if(accuracy_test_arr[-1] >= 0.95 or epochs > 51):
        break


#PLotting the Number of Epochs vs Error
plt.title("Number of Epochs vs Error")
plt.xlabel("Number of Epochs")
plt.ylabel("Error (MSE)")
plt.plot(range(len(error_train_arr)), error_train_arr, c = 'orange')
plt.plot(range(len(error_test_arr)), error_test_arr, c = 'green')
#plt.legend()
plt.show()


#PLotting the Number of Epochs vs Energy 
plt.title("Number of Epochs vs Energy")
plt.xlabel("Number of Epochs")
plt.ylabel("Energy")
plt.plot(range(len(mse_train_arr)), error_train_arr, c = 'orange')
plt.plot(range(len(mse_test_arr)), error_test_arr, c = 'green')
#plt.legend()
plt.show()        