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

#Initializing parameters
w_init = np.random.uniform(0, 1, size = (24, 1))
w_init_bias = np.random.uniform(0, 1, size = (24, 1))
w_final = np.random.uniform(0, 1, size = (24, 1))
w_final_bias = np.random.uniform(0, 1, size = (24, 1))
u = []                                                 #Induced local field from initial layer
z = []                                                 #Output from initial layer 
p = 0                                                  #Induced local field from final layer
y = 0                                                  #Output from final layer
learning_rate = 0.1

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
    temp = []
    for i in range(300):
        for j in range(24):
            
            
            
""" --- Main Algorithm --- """

while(1):
    y = feed_forward()