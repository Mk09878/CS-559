"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import time
#Defining the learning rate, initial weight, energy list 
learning_rate = 0.01
w0 = np.array([0.2, 0.5])
w0 = w0.reshape(2,1)
E_list = []
grad = np.empty((2,1))
H = np.empty((2,2))
steps = 0
w0_x = []
w0_y = []


start = time.time()
while(w0[0] + w0[1] < 1 and w0[0] > 0 and w0[1] > 0):
    
    w0_x.append(w0[0])
    w0_y.append(w0[1])
    grad[0] = (1 / (1 - w0[0] - w0[1])) - (1 / w0[0]) 
    grad[1] = (1 / (1 - w0[0] - w0[1])) - (1 / w0[1])
    
    change_in_weight = learning_rate * grad
    
    if np.linalg.norm(w0 - (w0 - change_in_weight)) < 0.001:
        break
    else:
        w0 = np.subtract(w0,change_in_weight)
    
    E = -np.log(1 - w0[0] - w0[1]) - np.log(w0[0]) - np.log(w0[1])
    E_list.append(E)
    steps += 1

print(time.time() - start)    
plt.title("Trajectory at each iteration")
plt.plot(w0_x, w0_y, c = 'black')
plt.xlabel("w0[0]")
plt.ylabel("w0[1]")
plt.show()

plt.title("Steps vs Energy")
plt.plot(range(steps), E_list, c = 'black')
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.show()    

print(steps)


w0 = np.array([0.2, 0.5])
w0 = w0.reshape(2,1)
E_list = []
steps = 0
w0_x = []
w0_y = []

start = time.time()
while(w0[0] + w0[1] < 1 and w0[0] > 0 and w0[1] > 0):
    
    w0_x.append(w0[0])
    w0_y.append(w0[1])
    grad[0] = (1 / (1 - w0[0] - w0[1])) - (1 / w0[0]) 
    grad[1] = (1 / (1 - w0[0] - w0[1])) - (1 / w0[1])
    
    H[0][0] = (1 / (1 - w0[0] - w0[1])**2) + (1 / w0[0] ** 2)
    H[0][1] = H[1][0] = (1 / (1 - w0[0] - w0[1])**2)
    H[1][1] = (1 / (1 - w0[0] - w0[1])**2) + (1 / w0[1] ** 2)
    
    change_in_weight = learning_rate * np.matmul(np.linalg.inv(H), grad)
    
    if np.linalg.norm(w0 - (w0 - change_in_weight)) < 0.001:
        break
    else:
        w0 = w0 - change_in_weight
    
    E = -np.log(1 - w0[0] - w0[1]) - np.log(w0[0]) - np.log(w0[1])
    E_list.append(E)
    steps += 1

print(time.time() - start)   
plt.title("Trajectory at each iteration")
plt.plot(w0_x, w0_y, c = 'black')
plt.xlabel("w0[0]")
plt.ylabel("w0[1]")
plt.show()

plt.title("Steps vs Energy")
plt.plot(range(steps), E_list, c = 'black')
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.show()    

print(steps)

  
    
    
    
    
