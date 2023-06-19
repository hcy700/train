
import torch
from collections import Counter
import torch.nn.functional as Fun
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv

strain = pd.read_csv('./data/recipes_train.csv')
stest = pd.read_csv('./data/recipes_test.csv')
x_train = strain.drop(columns=['id', "cuisine"]).values
y_train = strain["cuisine"].values
# x_train = strain.iloc[:, 2:]
# y_train = strain.iloc[:, 1]
sdata= pd.DataFrame()
sdata = strain['cuisine']
label = []

# 按照食谱产地进行分组,将文字标签转为数字标签
food_count = {} # 建立国家标签字典
for index,group in strain.groupby("cuisine"):
     location = group["cuisine"].head(1).item()
     food_count[location] = len(food_count.keys())
for country in sdata:
    label.append(food_count[country])
# strain['label'] = label # 添加数字标签列
# print(food_count)
n_data = len(label)
# print(n_data)
# print(label[0:5])
label = torch.tensor(label)
# 标签转化为独热编码
one_hot = np.zeros((n_data, 5))
for i in range(n_data):
    one_hot[i, label[i]] = 1.0
# print(one_hot)

# 预测数字标签转为文字标签
def get_key(val):
    for key, value in food_count.items():
         if val == value:
             return key
    return 0

# 分割训练集验证集
X_train, X_valid, y_train, y_valid = train_test_split(x_train, label, test_size=0.2, random_state=42)

# # 特征标准化
X_train = torch.from_numpy(X_train).to(torch.float32)
X_valid = torch.from_numpy(X_valid).to(torch.float32)
# X_train -= X_train.mean(axis=0)
# X_valid -= X_valid.mean(axis=0)
X_valid = X_valid.cuda()
X_train = X_train.cuda()
y_valid = y_valid.cuda()
y_train = y_train.cuda()
# print(X_valid.sum(axis=1))
# print(X_valid.shape)




#  定义模型 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # 定义隐藏层网络
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden//8)
        self.out = torch.nn.Linear(n_hidden//8, n_output)   # 定义输出层网络

    def forward(self, x):
        x = Fun.tanh(self.hidden1(x))
        x = Fun.tanh(self.hidden2(x))  # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        x = self.out(x)                # 输出层不用激活函数
        return x

# 定义优化器损失函数
net = Net(n_feature=383, n_hidden=128, n_output=5).cuda()   #n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
optimizer = torch.optim.SGD(net.parameters(), lr=0.001) # 优化器选用随机梯度下降方式
loss_func = torch.nn.CrossEntropyLoss() # 对于多分类一般采用的交叉熵损失函数,

# 4. 训练数据
from tqdm import tqdm
for t in tqdm(range(50000)):
    out = net(X_train)                 # 输入input,输出out
    loss = loss_func(out, y_train)     # 输出与label对比
    optimizer.zero_grad()   # 梯度清零
    loss.backward()         # 前馈操作
    optimizer.step()        # 使用梯度优化器
#print(accuracy = float((out == y_valid.data.numpy()).astype(int).sum()) / float(target_y.size))

# 5. 得出结果
out = net(X_valid) #out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
#print(out)
prediction = torch.max(out, 1)[1] # 返回index  0返回原值
pred_y = prediction.cpu().numpy()
target_y = y_valid.cpu().numpy()
#print(prediction)
#print(y_valid)
accuracy1 = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("预测测试准确率",accuracy1)

out = net(X_train) #out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
#print(out)
prediction = torch.max(out, 1)[1] # 返回index  0返回原值
pred_y = prediction.cpu().numpy()
target_y = y_train.cpu().numpy()
# 6.衡量准确率
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("预测训练准确率",accuracy)


