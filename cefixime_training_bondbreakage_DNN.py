# 导入模块
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='gbk')
    input_data = df.iloc[:, :78].values.astype('float32')
    output_data = df.iloc[:, 78].values.astype('float32')
    return input_data, output_data


# 创建神经网络模型
def create_neural_network(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(units=400, activation='tanh'),
        tf.keras.layers.Dense(units=200, activation='relu'),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=50, activation='sigmoid'),
        tf.keras.layers.Dense(units=1)
    ])
    return model


# 设置随机种子以便结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 加载数据并进行预处理（此处省略了预处理步骤，但您应该添加它）
CSV_FILE_PATH = "D:/pythonproject/Cefixime_feature_abstraction/Cefixime_data/Cefixime/cefixime_bondbreakage_alldata.csv"
input_data, output_data = load_data(CSV_FILE_PATH)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.1)

# 创建神经网络模型
input_shape = X_train.shape[1]
model = create_neural_network(input_shape)

# 定义模型保存路径和保存条件
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'save/model_checkpoint.h5')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_loss',
                                                      mode='min')

# 编译模型并进行训练
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=10000, batch_size=8, validation_split=0.1, callbacks=[model_checkpoint])

# 在测试集上评估模型性能并打印结果
y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
print(f'Test Mean Squared Error: {test_mse}')
print(f'Test Mean Absolute Error: {test_mae}')

# 绘制训练过程中的损失曲线并保存图像
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./loss_curve4.png')
plt.show()

# 将数据保存为CSV文件
data = {
    'epoch': list(range(1, len(history.history['loss']) + 1)),
    'loss': history.history['loss'],
    'val_loss': history.history['val_loss']
}

df1 = pd.DataFrame(data)
df1.to_csv('./loss_data4.csv', index=False)

#数据输出与保存
predicted_probabilities = y_pred   # 随机生成预测概率作为示例
true_labels = y_test  # 随机生成二分类真实标签作为示例

predicted_probabilities_1d = predicted_probabilities.ravel()

output_data = {'Predicted': predicted_probabilities_1d, 'True': true_labels}
df = pd.DataFrame(output_data)
csv_filename = 'predictions4.csv'
df.to_csv(csv_filename, index=False)

plt.figure(figsize=(10, 6))
plt.plot(range(len(df)), df['True'], color='red', label='True Labels', marker='x')
plt.plot(range(len(df)), df['Predicted'], color='blue', label='Predicted Probabilities', marker='o')
plt.title('Predicted vs True Labels')
plt.xlabel('Data Point Index')
plt.ylabel('Label/Probability')
plt.legend()
plt.xticks(range(len(df)))
plt.savefig('./prediction4')
plt.show()

# print(predicted_probabilities.shape)  # 如果它是numpy数组
# print(len(true_labels))  # 如果它是列表或其他可迭代对象
'''
# 将预测结果和真实标签写入CSV文件
output_data = {'Predicted': predicted_probabilities, 'True': true_labels}
df = pd.DataFrame(output_data)
csv_filename = 'predictions.csv'
df.to_csv(csv_filename, index=False)


plt.figure(figsize=(10, 6))
plt.plot(range(len(df)), df['True'], color='red', label='True Labels', marker='x')
plt.plot(range(len(df)), df['Predicted'], color='blue', label='Predicted Probabilities', marker='o')
plt.title('Predicted vs True Labels')
plt.xlabel('Data Point Index')
plt.ylabel('Label/Probability')
plt.legend()
plt.xticks(range(len(df)))
plt.savefig('./prediction')
plt.show()
'''
'''
# 加载数据
CSV_File_Path = "D:/pythonproject/Cefixime_feature_abstraction/Cefixime_data/Cefixime/cefixime_Habstraction_alldata.csv"
df = pd.read_csv(CSV_File_Path, encoding='gbk')  # 读取训练数据

# 数据分类
input_n1 = df.iloc[:, :40].values
output_n1 = df.iloc[:, 40].values

input_data_n1 = np.array(input_n1, dtype='float32')
output_data_n1 = np.array(output_n1, dtype='float32')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(input_data_n1, output_data_n1, test_size=0.1, random_state=42)

# 创建H相关神经网络的函数
def create_neural_network_1():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_data_n1.shape[1]),
        tf.keras.layers.Dense(units=200, activation='tanh'),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=50, activation='sigmoid'),  # 根据问题调整激活函数
        tf.keras.layers.Dense(units=1),
    ])
    return model

# 创建神经网络
network1 = create_neural_network_1()

# 定义模型保存路径和保存条件
checkpoint_filepath_n1 = os.path.join(os.getcwd(), 'save/model_checkpoint_n1.h5')

model_checkpoint_callback_n1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath_n1,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

# 编译模型
network1.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                 loss='mse',
                 metrics=['mae'])
# 加载最佳模型权重
#network1.load_weights(checkpoint_filepath_n1)

# 训练模型并添加保存权重的回调
history = network1.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=1,
    validation_split=0.1,
    callbacks=[model_checkpoint_callback_n1]
)

# 在测试集上评估模型性能
y_pred = network1.predict(X_test)
# y_pred_binary = np.round(y_pred)  # 将输出转换为二进制形式

# 计算均方误差和平均绝对误差
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)

print(f'Test Mean Squared Error: {test_mse}')
print(f'Test Mean Absolute Error: {test_mae}')

# 加载最佳模型权重
# network1.load_weights(checkpoint_filepath_n1)

# 绘制训练过程中的损失曲线
l1 = plt.plot(history.history['loss'], label="loss")
l2 = plt.plot(history.history['val_loss'], label="val_loss")
plt.legend(['loss','val_loss'])
plt.savefig('./loss1')
plt.show()
'''
'''
predicted_probabilities = y_pred   # 随机生成预测概率作为示例
true_labels = y_test  # 随机生成二分类真实标签作为示例

# 将预测结果和真实标签写入CSV文件
output_data = {'Predicted': predicted_probabilities, 'True': true_labels}
df = pd.DataFrame(output_data)
csv_filename = 'predictions.csv'
df.to_csv(csv_filename, index=False)


plt.figure(figsize=(10, 6))
plt.plot(range(len(df)), df['True'], color='red', label='True Labels', marker='x')
plt.plot(range(len(df)), df['Predicted'], color='blue', label='Predicted Probabilities', marker='o')
plt.title('Predicted vs True Labels')
plt.xlabel('Data Point Index')
plt.ylabel('Label/Probability')
plt.legend()
plt.xticks(range(len(df)))
plt.savefig('./prediction')
plt.show()
'''
'''
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'测试准确率: {accuracy}')
print(f'测试精确率: {precision}')
print(f'测试召回率: {recall}')

# 绘制训练过程中的损失曲线
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.show()

# 加载最佳模型权重
# network1.load_weights(checkpoint_filepath_n1)
'''


