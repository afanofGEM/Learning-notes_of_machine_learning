import numpy as np
import matplotlib.pyplot as plt
import math
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y = np.array([0, 0, 0, 1, 1, 1])     #(m,)

w = np.zeros_like(x[0])
b = 1.0

def sigmoid(x,w,b):
    z = np.dot(w,x) + b
    return 1/(1+np.exp(-z))

def cost_function(x,y,w,b):
    m,n = x.shape
    loss = 0.0
    for i in range(m):
        f = sigmoid(x[i],w,b)
        loss += y[i]*(-np.log(f)) + (1-y[i]) * (-np.log(1-f))
    return loss/m

cost =  cost_function(x,y,w,b)

def calculate_gradient(x,y,w,b):
    m,n = x.shape
    dJ_dw = np.zeros(n)
    dJ_db = 0.0

    for i in range(m):
        error = sigmoid(x[i],w,b) - y[i]
        dJ_db += error
        for j in range(n):
            dJ_dw[j] += error * x[i][j]

    dJ_db /= m
    dJ_dw /= m

    return dJ_dw,dJ_db

dJ_dw,dJ_db = calculate_gradient(x,y,w,b)

def gradient_descent(x,y,w,b,time,alpha):
    m,n = x.shape
    cost_h = []
    b_h = []
    w_h = []

    cost = cost_function(x,y,w,b)
    cost_h.append(cost)
    b_h.append(b)
    w_h.append(w)
    for i in range(time):
        dJ_dw , dJ_db = calculate_gradient(x,y,w,b)
        for j in range(n):
            w[j] -= alpha * dJ_dw[j]
        b -= alpha * dJ_db

        temp_cost = cost_function(x,y,w,b)
        
        if(temp_cost > cost):
            print("学习率选择错误")
            return cost_h,b_h,w_h,w,b  
        else: 
            b_h.append(b)
            w_h.append(w)
    return cost_h,b_h,w_h,w,b

time = 10000
alpha = 0.1
cost_h,b_h,w_h,w,b = gradient_descent(x,y,w,b,time,alpha)
print(w)
print(b)