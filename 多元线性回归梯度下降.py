# 输出预测值
import copy, math
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
alpha = 0.0000000001  # 10^-7 级别

def predict(w,b,x): 
    m = x.shape[0]
    y_pred = np.zeros(m)
    for i in range(m):
        y_pred[i] = np.dot(w, x[i]) + b
    return y_pred
# print(f'y_predict:{y_predict}')

# 代价函数并没有差异，因为都是预测与真实值的均方误差
def calculate_cost(w,b,x,y):
    y_p = predict(w,b,x)
    total_cost = np.sum((y-y_p)**2) / (2*len(y))  # 矩阵直接加减
    return total_cost

cost = calculate_cost(w_init,b_init,X_train,y_train)
# print(cost)

# 计算gradient
def calculate_gradient(w,b,x,y):
    y_p = predict(w,b,x)
    dJ_dw = np.zeros(len(w))
    dJ_db = 0.0
    m,n = x.shape

    for i in range(m): #公式最外层的累加
        error = y_p[i] - y[i] #值
        for j in range(n):  # 对于每一个样本，逐个累加各个特征的导数值
            dJ_dw[j] += error * x[i][j]
        dJ_db += error

    return dJ_dw/m, dJ_db/m
         

dJ_dw,dJ_db = calculate_gradient(w_init, b_init, X_train, y_train)
# print(dJ_dw)
# print(dJ_db)

def gradient_descent(w,b,x,y,time,alpha):
    m,n = x.shape
    Jh = []
    wh = []
    wb = []
    cost = calculate_cost(w,b,x,y)

    Jh.append(cost)
    wh.append(w.copy())
    wb.append(b)
    for i in range(time):
        dJ_dw,dJ_db = calculate_gradient(w, b, x, y)
        w -= alpha * dJ_dw
        b -= alpha * dJ_db
        temp_cost = calculate_cost(w,b,x,y)

        if temp_cost > cost:
            print('学习率选择错误')
            return Jh,wh,wb
        else:
            cost = temp_cost
            Jh.append(cost)
            wh.append(w.copy())
            wb.append(b)
    return Jh,wh,wb

Jh,wh,wb = gradient_descent(w_init,b_init,X_train,y_train,10000,alpha)
plt.plot(Jh,c='purple')
plt.xlabel('iterations')
plt.ylabel('the total cost')
plt.show()

plt.figure()
# wh是二维向量组，所以得分开画
wh = np.array(wh)
for i in range(wh.shape[1]):
    plt.plot(wh[:,i],label=f'the change of w{i}')
plt.plot(wb,c='blue',label='the change of b')
plt.xlabel('iterations')
plt.ylabel('the change of parameters')
plt.legend()
plt.show()