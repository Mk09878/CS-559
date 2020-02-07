"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)
#Intializing the weights and creating a weight vector
weight_vector = np.zeros((1,3))
w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
weight_vector[0][0] = w0
weight_vector[0][1] = w1
weight_vector[0][2] = w2

#Initializing the input vector S
S = np.zeros((100,2))
for x in range(100):
    S[x] = np.random.uniform(-1, 1, 2)

#Initializing S1 and S0
S0 = np.zeros((0,2))
S1 = np.zeros((0,2))
for x in S:
    temp = np.insert(x, 0, 1)
    if(np.dot(temp, weight_vector.T) >= 0):
        #print(np.dot(temp, weight_vector.T))
        x = x.reshape(1,2)
        S1 = np.concatenate((S1, x))
    else:
        x = x.reshape(1,2)
        S0 = np.concatenate((S0, x))


#Plotting the graph
X = np.linspace(-1, 1)
Y = - (X*w1 + w0) / w2
plt.ylim(-1,1)
plt.title("Plot")
plt.scatter(*zip(*S0), label = "S0", marker = 'o')
plt.scatter(*zip(*S1), label = "S1", marker = 'P')
plt.plot(X, Y, c = 'black', label='Separator')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Perceptron Training Algorithm
learning_rate = 1
weight_vector_pta = np.zeros((1,3))
w0_pta = np.random.uniform(-1, 1)
w1_pta = np.random.uniform(-1, 1)
w2_pta = np.random.uniform(-1, 1)
weight_vector_pta[0][0] = w0_pta
weight_vector_pta[0][1] = w1_pta
weight_vector_pta[0][2] = w2_pta
epoch = 0
missclass_arr = []
no_of_errors = 0
while(epoch == 0 or no_of_errors != 0):
    epoch += 1
    no_of_errors = 0
    for x in S:
        temp = np.insert(x, 0, 1)
        if(np.dot(temp, weight_vector_pta.T) >= 0):
            if(x in S1):
                #Correct prediction
                continue
            else:
                #Error in prediction
                #Update weight and increment no_of_errors
                weight_vector_pta -= learning_rate * temp 
                no_of_errors += 1
                
        else:
            if(x in S0):
                #Correct prediction
                continue
            else:
                #Error in prediction
                #Update weight and increment no_of_errors
                weight_vector_pta += learning_rate * temp
                no_of_errors += 1
    
    missclass_arr.append(no_of_errors)

#Plotting the graph (from scratch)
X = np.linspace(-1, 1)
Y = - (X*weight_vector_pta[0][1] + weight_vector_pta[0][0]) / weight_vector_pta[0][2]
plt.ylim(-1,1)
plt.title("Plot (from scratch)")
plt.scatter(*zip(*S0), label = "S0", marker = 'o')
plt.scatter(*zip(*S1), label = "S1", marker = 'P')
plt.plot(X, Y, c = 'black', label='Separator')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Plotting the graph of Epoch Number vs Number of Misclassifications
epoch_arr = np.arange(0, epoch)
#plt.ylim(-1,1)
plt.title("Epoch Number vs Number of Misclassifications")
plt.plot(epoch_arr, missclass_arr, c = 'black')
plt.xlabel("Epoch Number")
plt.ylabel("Number of Misclassifications")
plt.show()
