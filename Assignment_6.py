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

def polynomial_kernel(x, y, d = 5):
    return (1 + np.dot(x, y)) ** d

def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x-y)**2 / (sigma ** 2))
        
#Creating the kernel matrix
K = [[0 for x in range(100)] for y in range(100)]
for i in range(100):
    for j in range(100):
        K[i][j] = polynomial_kernel(x[i], x[j], 5)

temp = np.ones(100) * -1
P = cvxopt.matrix(np.outer(d,d) * K)
q = cvxopt.matrix(temp)
G = cvxopt.matrix(np.diag(temp))
h = cvxopt.matrix(np.zeros(100))
A = cvxopt.matrix(d, (1,100))
b = cvxopt.matrix(0.0)

#result = cvxopt.solvers.qp(P, q, G, h, A, b)
result = cvxopt.solvers.qp(cvxopt.matrix(np.outer(d,d) * K), 
                           cvxopt.matrix(temp), cvxopt.matrix(np.diag(temp)), 
                           cvxopt.matrix(np.zeros(100)), 
                           cvxopt.matrix(d, (1,100)), 
                           cvxopt.matrix(0.0))

#alpha = np.ravel(result['x'])
alpha = np.asarray(result['x']).flatten()
sv_cp1_x = []
sv_cp1_y = []
sv_cn1_x = []
sv_cn1_y = []
sv_x = []
sv_y = []
indices = []
for i in range(100):
    if(alpha[i] > 1e-5):
        """if(d[i] == 1):
            sv_cp1_x.append(x[i])
            sv_cp1_y.append(d[i])
        else:
            sv_cn1_x.append(x[i])
            sv_cn1_y.append(d[i])"""
        sv_x.append(x[i])
        sv_y.append(d[i])
        indices.append(i)

#sv_x = sv_cp1_x + sv_cn1_x
#sv_y = sv_cp1_y + sv_cn1_y

summation = 0
g = 0
hyperplane = []
p_hyperplane = []
n_hyperplane = []
alpha_sv = []
d_sv = []
for i in indices:
    alpha_sv.append(alpha[i])
    d_sv.append(d[i])
for i in range(len(sv_x)):
    summation += alpha_sv[i] * d_sv[i] * polynomial_kernel(sv_x[i], sv_x[1], 5)
bias = sv_y[1] - summation
print(bias)


x_coord = np.linspace(0.0, 1.0, num=1000)
y_coord = np.linspace(0.0, 1.0, num=1000)

for i in range(len(x_coord)):
    print("Iteration:",i)
    for j in range(len(y_coord)):
        g = 0
        for k in range(len(sv_x)):
            g += alpha_sv[k] * d_sv[k] * polynomial_kernel(sv_x[k], np.asarray([x_coord[i], y_coord[j]]), 5)
        g = g + bias
        
        if -0.1 < g < 0.1:
            hyperplane.append([x_coord[i], y_coord[j]])
        elif 0.9 < g < 1.1:
            p_hyperplane.append([x_coord[i], y_coord[j]])
        elif -1.1 < g < -0.9:
            n_hyperplane.append([x_coord[i], y_coord[j]])
    

fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(*zip(*class_one), c = 'red', label = 'Class 1')
plt.scatter(*zip(*class_minusone), c = 'green', label = 'Class -1')
plt.scatter(*zip(*p_hyperplane), c = 'red',s=1, label = 'Hyperplane 1')
plt.scatter(*zip(*hyperplane), c = 'blue',s=1, label = 'Margin')
plt.scatter(*zip(*n_hyperplane), c = 'green', s=1, label = 'Hyperplane -1')
plt.scatter(*zip(*sv_x), facecolors = 'none', edgecolors='black',label='Support Vectors')
plt.legend(loc = 'best')
plt.show()        