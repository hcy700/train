import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from pandas.plotting import scatter_matrix
import matplotlib as mpl

strain = pd.read_csv('./data/recipes_train.csv')
stest = pd.read_csv('./data/recipes_test.csv')



#print(strain[0])
#strain.info()
#print(strain.head(8))
# ##标签分布可视化分析
# plt.figure(figsize=(10,10))
# strain.cuisine.value_counts().plot(kind='bar')
# plt.show()

# #特征稀疏性可视化,可视化chinese食材分布
# sdata= pd.DataFrame()
# sdata = strain[strain.cuisine =='chinese']
# x = sdata.iloc[:,2:]
# col = x.sum(axis=0)
# value = col.values.tolist()
# # print(col.iloc[0])
# # print(type(col))
# plt.figure()
# subplot_count = 5
# plt.bar(range(len(value)), value)
#
# # colors = ['white', 'blue']
# # cmap = mpl.colors.ListedColormap(colors)
# # plt.imshow(x, cmap=cmap)
# plt.show()




# #标签与特征相关性可视化
# x = strain.iloc[:,2:]
# y = strain.iloc[:,1]
# print(x.shape)
# pca = PCA().fit(x)
# plt.figure(figsize=(20,5))
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('降维后的特征数目')
# plt.ylabel('累计可解释性方差比例之和')
# plt.show()

# 计算余弦相似度
# 按照食谱产地进行分组统计
food_count = {}
for _,group in strain.groupby("cuisine"):
    location = group["cuisine"].head(1).item()
    food_count[location] = {}
    for col in group.columns:
        if col not in ["id", "cuisine"]:
            food_count[location][col] = group[col].sum()

dist=[]
for i in range(5):
    location = list(food_count.keys())[i]
    dist.append(list(food_count[location].values()))
#dist = np.array(dist)
cosvalue = np.zeros((5,5))
for i in range(5):
    for j in range(i,5):
        cosvalue[i][j] = np.dot(dist[i],dist[j])/(np.linalg.norm(dist[i]) * np.linalg.norm(dist[j]))
print(cosvalue)
sns.heatmap(cosvalue, cmap='Blues', annot=True)
plt.show()