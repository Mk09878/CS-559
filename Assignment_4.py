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
plt.scatter(x, d, c = 'black')
plt.xlabel("X")
plt.ylabel("D")
plt.show()

