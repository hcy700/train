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

def loaddata():
    strain = pd.read_csv('./data/recipes_train.csv')
    stest = pd.read_csv('./data/recipes_test.csv')
    x_train = strain.drop(columns=['id', "cuisine"]).values
    y_train = strain["cuisine"].values
    # x_test = stest.iloc[:, 1:]
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # print(X_test.shape)
    # y_test = np.zeros((X_test.shape[0], 1))
    # print(y_test.shape)
    # dt = tree.DecisionTreeClassifier()
    # dt.fit(X_train, y_train)
    # dt.get_params()
    # score = dt.score(X_train, y_train)
    # print("此模型得分为%s" % score)
    # print(dt.predict(X_test))
    return X_train, X_valid, y_train, y_valid


import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
def bagging(n,x,y,x_test,y_test, cnt=5):  # n为取样数
    prediction = []
    mid = []
    result = []

    for i in range(cnt):   # 默认3个基学习器
        dt = tree.DecisionTreeClassifier(max_depth=15)
        #dt = LogisticRegression(penalty="l1", C=0.5, solver="liblinear")
        x_train = []
        y_train = []
        for j in range(n):  # 随机取样数量

            a = np.random.randint(0, len(x) - 1)
            x_train.append(x[a, :])
            y_train.append(y[a])
        dt.fit(x_train, y_train)
        # dt.get_params()
        prediction.append(dt.predict(x_test))
        #prediction = np.array(prediction)
    for index in range(len(x_test)):
        mid.append([x[index] for x in prediction])
        count = Counter(mid[index])
        result.append(max(count))
    #score = dt.score(x_test, y_test)
    # print(result[0:5])
    # print(y_test[0:5])
    # print(dt.score(x_train, y_train))
    score = accuracy_score(result, y_test)
    f1 = f1_score(y_test,result,average='macro')
    return score,f1

X_train, X_valid, y_train, y_valid = loaddata()
# # 按照食谱产地进行分组,将文字标签转为数字标签
# food_count = {} # 建立国家标签字典
# for index,group in X_train.groupby("cuisine"):
#      location = group["cuisine"].head(1).item()
#      food_count[location] = len(food_count.keys())
# for country in sdata:
#     label.append(food_count[country])
y_valid = y_valid.tolist()
score,f1 = bagging(1000,X_train,y_train,X_valid,y_valid)
print(score)
print(f1)

import tqdm
scores = []
for i in tqdm.tqdm(range(1, 31, 2)):
    scores.append(bagging(1000,X_train,y_train,X_valid,y_valid, i))

plt.plot(scores)
plt.show()