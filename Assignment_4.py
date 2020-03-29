"""
Written By: Mihir Kelkar
Date: 3/25/2020
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)

#Initializing x, v, d
x = np.random.uniform(0, 1, size = (300, 1))
v = np.random.uniform(-0.1, 0.1, size = (300, 1))
d = (np.sin(20 * x) + (3 * x) + v)

#PLotting x vs d
plt.title("x vs d")
plt.xlabel("X")
plt.ylabel("D")
plt.scatter(x, d, c = 'black')
plt.show()

#Initializing parameters
w_init = np.random.uniform(0, 1, size = (24, 1))
w_init_bias = np.random.uniform(0, 1, size = (24, 1))
w_final = np.random.uniform(0, 1, size = (24, 1))
w_final_bias = np.random.uniform(0, 1, size = (1, 1))
induced_local_field_init = [0] * 24                                                 #Induced local field from initial layer
output_init = [0] * 24                                                 #Output from initial layer 
induced_local_field_final = 0                                                 #Induced local field from final layer
output_final = 0
output_final_arr = [0] * 300                                                 #Output from final layer
learning_rate = 8
error = 0
error_arr = []
epochs = 0
    
""" --- Activation Functions --- """

#Feedforward
def ff_init_act(u): 
    return np.tanh(u)

def ff_final_act(y):
    return y

#Feedback
def fb_init_act(u):
    return (1 - np.tanh(u) ** 2)

def fb_final_act(y):
    return 1


""" --- Feedforward Training --- """
def feed_forward():
    global output_final
    global induced_local_field_final
    
    #Calculating the local fields and the outputs
    for j in range(24):
        induced_local_field_init[j] = w_init[j] * x[i] + w_init_bias[j] 
        output_init[j] = ff_init_act(induced_local_field_init[j])
    
    induced_local_field_final = np.matmul(np.array(output_init).T,w_final) + w_final_bias
    output_final = ff_final_act(induced_local_field_final)
    output_final_arr[i] = output_final
    #print(output_final)
    
            
""" --- Backpropagation --- """
def feed_back():
    global output_final
    global induced_local_field_final
    global w_init
    global w_init_bias
    global w_final
    global w_final_bias
    grad_init = np.zeros((24, 1))
    grad_init_bias = np.zeros((24, 1))
    grad_final = np.zeros((24, 1))
    count = 0
    #Calculating the gradients
    grad_final_bias = -1 * fb_final_act(induced_local_field_final) * (d[i] - output_final)
    for j in range(24):
        count+=1
        grad_final[j] = -1 * output_init[j] * fb_final_act(induced_local_field_final) * (d[i] - output_final)
        grad_init_bias[j] = -1 * fb_init_act(induced_local_field_init[j]) * w_final[j] * fb_final_act(induced_local_field_final) * (d[i] - output_final)
        grad_init[j] = -1 * x[i] * fb_init_act(induced_local_field_init[j]) * w_final[j] * fb_final_act(induced_local_field_final) * (d[i] - output_final)
    
    #Updating the weights
    w_init = w_init - learning_rate * (grad_init/300)
    w_init_bias = w_init_bias - learning_rate * (grad_init_bias/300)
    w_final = w_final - learning_rate * (grad_final/300)
    w_final_bias = w_final_bias - learning_rate * (grad_final_bias/300) 

    
""" --- Main Algorithm --- """
while(1):
    for i in range(300):
        feed_forward()
        feed_back()
        error += (d[i] - output_final) ** 2
        
    
    error = error / 300
    error_arr.append(error[0][0])
    print(error)
    
    #Decreasing learning rate if no improvements take place
    if(error_arr[epochs] > error_arr[epochs - 1]):
        learning_rate *= 0.9
    
    #Terminate if error goes below some threshold
    if(error < 0.01):
        feed_back()
        break
    
    epochs += 1

#PLotting the Number of Epochs vs Error (MSE)
plt.title("Number of Epochs vs Error (MSE)")
plt.xlabel("Number of Epochs")
plt.ylabel("Error (MSE)")
plt.plot(range(len(error_arr)), error_arr, c = 'black')
plt.show()

#PLotting 
plt.title("x vs d")
plt.xlabel("X")
plt.ylabel("D")
plt.scatter(x, d, c = 'red', label = 'Desired')
plt.scatter(x, output_final_arr, c = 'blue', label = 'Predicted')
plt.legend()
plt.show()
    
    
    