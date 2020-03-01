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
y = np.ones((50,1))
for i in range(50):
    y[i] = i + 1 + np.random.uniform(-1, 1)


temp = np.array([np.ones(50), x])
temp = np.transpose(np.array([np.ones(50), x]))