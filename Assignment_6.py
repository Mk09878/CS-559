"""
Written By: Mihir Kelkar
Date: 4/18/2020
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxopt

np.random.seed(100)

#Defining the required variables
x = np.random.uniform(0, 1, (100, 2))
d = np.zeros(100)
class_one = []
class_minusone = []
sv_x = []
sv_d = []
sv_alpha = []
summation = 0
g = 0
hyperplane = []
p_hyperplane = []
n_hyperplane = []
x_coord = np.linspace(0.0, 1.0, num=1000)
y_coord = np.linspace(0.0, 1.0, num=1000)

#Assigning the input patterns to their respective classes
for i in range(len(x)):
    if(x[i][1] < 1/5 * np.sin(10 * x[i][0]) + 0.3 or (x[i][1] - 0.8)**2 + (x[i][0] - 0.5)**2 < 0.15**2):
        d[i] = 1
        class_one.append(x[i])
    else:
        d[i] = -1
        class_minusone.append(x[i])

class_one = np.asarray(class_one)
class_minusone = np.asarray(class_minusone)        

#PLotting the input patterns
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Input patterns")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.scatter(class_one[:, 0], class_one[:, 1], c = 'orange', label = "C1", marker = "P")
plt.scatter(class_minusone[:, 0], class_minusone[:, 1], c = 'green', label = "C-1", marker = 0)
plt.legend()
plt.show()

""" --- Kernels --- """

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, d = 5):
    return (1 + np.dot(x, y)) ** d

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (sigma ** 2))
        
#Creating the kernel matrix
K = [[0 for x in range(100)] for y in range(100)]
for i in range(100):
    for j in range(100):
        K[i][j] = polynomial_kernel(x[i], x[j], 5)

#Calculating the alpha values
temp = np.ones(100) * -1

result = cvxopt.solvers.qp(cvxopt.matrix(np.outer(d,d) * K), 
                           cvxopt.matrix(temp), cvxopt.matrix(np.diag(temp)), 
                           cvxopt.matrix(np.zeros(100)), 
                           cvxopt.matrix(d, (1,100)), 
                           cvxopt.matrix(0.0))

alpha = np.asarray(result['x']).flatten()

#Creating new lists based on support vectors
for i in range(100):
    if(alpha[i] > 1e-5):
        sv_x.append(x[i])
        sv_d.append(d[i])
        sv_alpha.append(alpha[i])

#Finding the bias        
for i in range(len(sv_x)):
    summation += sv_alpha[i] * sv_d[i] * polynomial_kernel(sv_x[i], sv_x[0], 5)
bias = sv_d[0] - summation

#Creating the hyperplanes
for i in range(len(x_coord)):
    print("Iteration:",i)
    for j in range(len(y_coord)):
        g = 0
        for k in range(len(sv_x)):
            g += sv_alpha[k] * sv_d[k] * polynomial_kernel(sv_x[k], np.asarray([x_coord[i], y_coord[j]]), 5)
        g = g + bias
        
        if (-0.1 < g < 0.1):
            hyperplane.append([x_coord[i], y_coord[j]])
        elif (0.9 < g < 1.1):
            p_hyperplane.append([x_coord[i], y_coord[j]])
        elif (-1.1 < g < -0.9):
            n_hyperplane.append([x_coord[i], y_coord[j]])
    

hyperplane = np.asarray(hyperplane)
p_hyperplane = np.asarray(p_hyperplane)
n_hyperplane = np.asarray(n_hyperplane)
sv_x = np.asarray(sv_x)

#Plotting the input patterns with hyperplanes
fig, ax = plt.subplots(figsize=(10,10))
plt.title("Input patterns with hyperplanes")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.scatter(class_one[:, 0], class_one[:, 1], c = 'orange', label = "C1", marker = "P")
plt.scatter(class_minusone[:, 0], class_minusone[:, 1], c = 'green', label = "C-1", marker = 0)
plt.scatter(p_hyperplane[:, 0], p_hyperplane[:, 1], c = 'orange',s=1, label = 'Hyperplane 1')
plt.scatter(hyperplane[:, 0], hyperplane[:, 1], c = 'red',s=1, label = 'Margin')
plt.scatter(n_hyperplane[:, 0], n_hyperplane[:, 1], c = 'green', s=1, label = 'Hyperplane -1')
plt.scatter(sv_x[:, 0], sv_x[:, 1], c = 'black', alpha = 0.8, label='Support Vectors')
plt.legend()
plt.show()   