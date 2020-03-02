"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(100)
#Initialize x and y
x = np.array([np.arange(1, 51)])
x = x.reshape(50,)
y = np.ones((50,))
for i in range(50):
    y[i] = i + 1 + np.random.uniform(-1, 1)

#Calculating the weight (w = y * x^T * (x * x^T)^-1 )
xones = np.vstack((np.ones((50,)), x))

trans = (np.vstack((np.ones((50,)), x))).T

inverse = np.linalg.inv(np.matmul(xones, xones.T))

temp = np.matmul(xones.T, inverse)

w = np.matmul(y, temp)
w = w.reshape(2,1)
#w = np.matmul(y, np.matmul((np.vstack((np.ones((50,)), x))).T, np.linalg.inv(np.matmul(np.vstack((np.ones((50,)), x)), (np.vstack((np.ones((50,)), x))).T))))

#PLotting the Linear Least Square Fit
plt.title("Linear Least Square Fit")
plt.plot(x, w[1] * x + w[0], c = 'black')
plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


learning_rate = 0.00001
w0 = np.array([0.2, 0.5])
w0 = w0.reshape(2,1)
grad = np.empty((2,1))
steps = 0
sum_0 = 0
sum_1 = 0
while(True):
    
    sum_0 = 0
    sum_1 = 0
    for i in range(50):
        sum_0 += (y[i] - (w0[0] + w0[1] * x[i])) * (-2)
    grad[0] = sum_0
    for j in range(50):
        sum_1 += (y[j] - (w0[0] + w0[1] * x[j])) * x[j] * (-2)
    grad[1] = sum_1
        
    change_in_weight = learning_rate * grad
    
    if np.linalg.norm(w0 - (w0 - change_in_weight)) < 0.00001:
        break
    else:
        w0 = np.subtract(w0,change_in_weight)
    
    steps += 1

difference = abs(w0 - w)