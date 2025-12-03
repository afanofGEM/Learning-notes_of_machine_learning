# 练习梯度下降算法

import math, copy
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,430,630, 730,])
m = len(x_train)
time = 1000
alpha = 0.1
w = 0
b = 0
w_history = []
b_history = []
J_history = []

def calculate_gradient(x_train, y_train, w_now, b_now, m):
    
    dJ_dw = np.dot(x_train, w_now*x_train + b_now - y_train) / m  # np.dot就是向量的点乘
    dJ_db = np.sum(w_now*x_train + b_now - y_train) / m  # np.sum累加向量
    return dJ_dw, dJ_db  # 可以返回两个

def calculate_cost(x_train, y_train, m, w, b):
    cost = 0
    for i in range(m):
        cost += (w* x_train[i] + b - y_train[i])**2
    cost /= (2*m)
    return cost
    
def gradient_descent(time, alpha, w_origin, b_origin, m, x_train, y_train, J_history, w_history, b_history):

    cost = calculate_cost(x_train, y_train, m, w_origin, b_origin)
    J_history.append(cost)
    w_history.append(w_origin)
    b_history.append(b_origin)
    
    for i in range(time):
        dJ_dw , dJ_db = calculate_gradient(x_train, y_train, w_origin, b_origin, m)
        w_origin -= alpha * dJ_dw
        b_origin -= alpha * dJ_db
        temp_cost = calculate_cost(x_train, y_train, m, w_origin, b_origin)
        if(temp_cost > cost):
            print('学习率选择有问题，代价函数无法收敛')
            return w_origin, b_origin, J_history, w_history, b_history
        else:
            cost = temp_cost
            J_history.append(cost)
            w_history.append(w_origin)
            b_history.append(b_origin)

    return w_origin, b_origin, J_history, w_history, b_history


w, b, J_history, w_history, b_history = gradient_descent(time, alpha, w, b, m, x_train, y_train, J_history, w_history, b_history)
print(f'w,b:{w},{b}')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.plot(J_history[:100])
plt.show()

plt.xlabel('iteration')
plt.ylabel('w and b')
plt.plot(w_history[:100])
plt.plot(b_history[:100])
plt.show()
    