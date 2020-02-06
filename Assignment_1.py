"""
Written By: Mihir Kelkar
Date: 2/6/2020
"""

#Importing the libraries
import numpy as np

#Intializing the weights
w0 = np.random.uniform(-0.25, 0.25)
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)

#Initializing the input vector S
S = np.zeros((100,2))
for x in range(100):
    S[x] = np.random.uniform(-1, 1, 2)

