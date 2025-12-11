# 创建neuron network的模版:定义结构，编译（损失函数，定义学习率，拟合数据）
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),  #表示每个样本的特征数为2，但是不清楚有多少个样本
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(2, activation='sigmoid', name = 'layer2'),
        Dense(1, activation='sigmoid', name = 'layer3')
     ]
)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(), # 二元交叉熵函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    x_train,y_train,          
    epochs=10, # 模型被训练10次
)

w1,b1 = model.get_layer('layer1').get_weights()
# print(w1,b1)
w2,b2 = model.get_layer('layer2').get_weights()
# print(w2,b2)
w3,b3 = model.get_layer('layer3').get_weights()
# print(w3,b3)

y_pre = model.predict(xn)