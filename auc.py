from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression

# 正负样本对以中国为例
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
# print(food_count)
# print(type(label))
# print(label[0:5])
# print(sdata[0:5])

# 训练
X_train, X_valid, y_train, y_valid = train_test_split(x_train, label, test_size=0.2, random_state=42)
print(y_valid[0:5])
dt = tree.DecisionTreeClassifier(max_depth=30)
dt.fit(X_train, y_train)
dt.get_params()

#存储概率值
pro = dt.predict_proba(X_valid)
pre = []
for i in range(pro.shape[0]):
    pre.append(pro[i][0])
# print(pro[0:5])
# print(pre[0:5])
score = accuracy_score(dt.predict(X_valid), y_valid)
print("此模型验证得分为%s" % score)
# print(dt.predict(X_valid)[0:5])

# # 计算auc值
pos = []
neg = []
auc = 0
for index in range(len(y_valid)):
    if y_valid[index] == 0:
        pos.append(index)
    else:
        neg.append(index)
        pre[index] = 1 - pre[index]
for i in pos:
    for j in neg:
        if pre[i] > pre[j]:
            auc += 1
        elif pre[i] == pre[j]:
            auc += 0.5
auc = auc * 1.0 / (len(pos)*len(neg))
print(auc)