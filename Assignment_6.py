"""
Written By: Mihir Kelkar
Date: 4/18/2020
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxopt

np.random.seed(100)

#Initializing the input patterns and the desired output
x = np.random.uniform(0, 1, (100, 2))
d = np.zeros(100)
class_one = []
class_minusone = []
for i in range(len(x)):
    if(x[i][1] < 1/5 * np.sin(10 * x[i][0]) + 0.3 or (x[i][1] - 0.8)**2 + (x[i][0] - 0.5)**2 < 0.15**2):
        d[i] = 1
        class_one.append(x[i])
    else:
        d[i] = -1
        class_minusone.append(x[i])

class_one = np.asarray(class_one)
class_minusone = np.asarray(class_minusone)        

#PLotting the 
plt.title("")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.scatter(class_one[:, 0], class_one[:, 1], c = 'orange', label = "C1")
plt.scatter(class_minusone[:, 0], class_minusone[:, 1], c = 'green', label = "C-1")
plt.legend()
plt.show()

""" --- Kernels --- """

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, d):
    return (1 + np.dot(x, y)) ** d

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (sigma ** 2))
        
#Creating the kernel matrix
k = [[0 for x in range(100)] for y in range(100)]
for i in range(100):
    for j in range(100):
        k[i][j] = polynomial_kernel(x[i], x[j], 5)

temp = np.ones(100) * -1
P = cvxopt.matrix(np.outer(d,d) * k)
q = cvxopt.matrix(temp)
G = cvxopt.matrix(np.diag(temp))
h = cvxopt.matrix(np.zeros(100))
A = cvxopt.matrix(d, (1,100))
b = cvxopt.matrix(0.0)

#result = cvxopt.solvers.qp(P, q, G, h, A, b)
result = cvxopt.solvers.qp(cvxopt.matrix(np.outer(d,d) * k), 
                           cvxopt.matrix(temp), cvxopt.matrix(np.diag(temp)), 
                           cvxopt.matrix(np.zeros(100)), 
                           cvxopt.matrix(d, (1,100)), 
                           cvxopt.matrix(0.0))

#alpha = np.ravel(result['x'])
alpha = np.asarray(result['x']).flatten()