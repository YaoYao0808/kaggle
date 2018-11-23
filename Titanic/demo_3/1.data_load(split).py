"""
数据划分思路：
用train.csv中的全体数据测试模型
任选部分train.csv中的数据测试模型，使用auc/精确度评价指标，选择评价较好的模型做最终结果预测
"""
# 设置结果显示完整

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data_train = pd.read_csv("../data/train.csv")

X = data_train.drop('Survived', 1)
y = data_train['Survived']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.33, random_state=42)

print(len(X_test_val)) #295

#用上述X_test_val，y_test_val来测试数据集并求评价指标





# 将data_train 分为data_train,data_train_val
# data_train_val用来验证数据模型，求出评价指标
# print(data_train.info())  #PassengerId

# #获取随机数
# random_num=random.sample(range(1,891),200)
# print(random_num[2])
#
# # data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
# data_train_val=np.array().reshape(200,12)
# print(data_train_val['PassengerId'].dtype) #int64

# for i in range(len(random_num)):
#     for j in range(len(data_train)):
#         if data_train_val['PassengerId'] == random_num[i]:
#             data_train_val.append



# data_train_val[data_train_val['PassengerId']==random_num]
#
# print(data_train_val.head(10))







