# 利用sklearn实现多元线性回归
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
import matplotlib.pyplot as plt

# linearregression 通常是基于正规方程的闭式解），一次性计算最优参数
# SGDRegressor ：gradient descent
x_train, y_train = load_house_data()
x_features = ['size(sqft)','bedrooms','floors','age']

# scaler = StandardScaler() scikit-learn 中用于数据标准化的类
# SGDRegressor是使用随机梯度下降算法训练的线性回归模型
scaler = StandardScaler()
x_std_norm = scaler.fit_transform(x_train)
# print(x_train[:,0])
# print(x_std_norm[:,0])

time = 10000
sdgr = SGDRegressor(max_iter = time)
sdgr.fit(x_std_norm, y_train)

# sdgr:
print(f"the iteration time: {sdgr.n_iter_}")
print(f"the parameter changing times: {sdgr.t_}")

b_norm = sgdr.intercept_  # 线性模型截距项
w_norm = sgdr.coef_  # 线性模型的参数项
print(f"the value of parameter w: {w_norm}")
print(f"the value of parameter b: {b_norm}")

# predict
y_pred = sdgr.predict(x_std_norm)
print(y_train[:10])
print(y_pred[:10])

col = x_train.shape[1]
whole_img, part_array = plt.subplots(1,col,sharey = True,figsize=(12,3))  # sharey允许共用y轴
for i in range(col):
    part_array[i].scatter(x_train[:,i],y_train,label='actual',c='red')
    part_array[i].set_xlabel(x_features[i])
    part_array[i].scatter(x_train[:,i],y_pred,label='predict',c='purple')
    part_array[i].legend()
part_array[0].set_ylabel('price')