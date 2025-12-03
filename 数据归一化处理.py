# 数据的归一化处理
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(32)
x = np.zeros((1000,10))
for i in range(x.shape[1]):
    x[:,i] = np.random.randint(1,10000,size=x.shape[0]) 
    # np.random.randint产生的只有一个数，所以要用size扩充

# print(x[:10])
x_mean_col = x.mean(axis=0) #纵向求平均
x_mean_row = x.mean(axis=1)
# print(x_mean_col.shape,x_mean_row.shape)

x_range_col = x.max(axis=0) - x.min(axis=0)
# print(x_range_col.shape)
x_std_col = x.std(axis=0)

# 均值归一化
# NumPy 发现维度不一致时，
# 会自动扩展（broadcast）较小的维度（只有1才会扩展），使它们形状兼容
x_mean_norm = (x-x_mean_col)/x_range_col  # 严格规定范围-1，1
# print(x_mean_norm[:10])
x_std_norm = (x-x_mean_col)/x_std_col #改变数据分布，不易受极端值影响

plt.figure(figsize=(15, 4))

# 原始数据散点图
plt.subplot(1, 3, 1)
plt.scatter(x[:, 0], x[:, 1], s=8,c='purple')
plt.title("Original")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 均值归一化
plt.subplot(1, 3, 2)
plt.scatter(x_mean_norm[:, 0], x_mean_norm[:, 1], s=8,c='purple')
plt.title("Mean Normalized")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# 标准差归一化
plt.subplot(1, 3, 3)
plt.scatter(x_std_norm[:, 0], x_std_norm[:, 1], s=8,c='purple')
plt.title("Std Normalized")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.tight_layout()
plt.show()