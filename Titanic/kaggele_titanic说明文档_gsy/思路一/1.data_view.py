# 1.对数据进行统计并可视化
import pandas as pd
import numpy as np

# 设置结果显示完整
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data_train=pd.read_csv("./data/train.csv")
print(data_train.info())#查看数据缺失情况
print(data_train.describe())#查看数据基本统计信息

'''
这艘船共有891个乘客
年龄(Age)存在缺失值，只有714个
客舱(Cabin)只有204个数据
Embarked缺失两个数据
'''

#使用matplotlib进行数据可视化

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data_train = pd.read_csv("./data/train.csv")
# 查看各乘客等级的获救情况
survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({'获救': survived_1, '未获救': survived_0})
df.plot(kind='bar', stacked=True)
plt.title('各船舱乘客获救情况')
plt.xlabel('船舱等级')
plt.ylabel('人数')

# 查看性别的获救情况
survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df2 = pd.DataFrame({'获救': survived_1, '未获救': survived_0})
df2.plot(kind='bar', stacked=True)
plt.title('男女乘客获救情况')
plt.xlabel('性别')
plt.ylabel('人数')

# #查看各船舱级别下的男女获救情况(方法一)
# fig=figure()
# fig.add_subplot(131)
# survived_0_one=data_train.Sex[data_train.Survived==0][data_train.Pclass==1].value_counts()
# survived_1_one=data_train.Sex[data_train.Survived==1][data_train.Pclass==1].value_counts()
# df2=pd.DataFrame({'获救':survived_1_one,'未获救':survived_0_one})
# df2.plot(kind='bar',stacked=True)
# plt.title('头等舱')
# plt.xlabel('性别')
# plt.ylabel('人数')
# fig.add_subplot(132)
# survived_0_two=data_train.Sex[data_train.Survived==0][data_train.Pclass==2].value_counts()
# survived_1_two=data_train.Sex[data_train.Survived==1][data_train.Pclass==2].value_counts()
# df4=pd.DataFrame({'获救':survived_1_two,'未获救':survived_0_two})
# df4.plot(kind='bar',stacked=True)
# plt.title('二等舱')
# plt.xlabel('性别')
# plt.ylabel('人数')
# fig.add_subplot(133)
# survived_0_three=data_train.Sex[data_train.Survived==0][data_train.Pclass==3].value_counts()
# survived_1_three=data_train.Sex[data_train.Survived==1][data_train.Pclass==3].value_counts()
# df5=pd.DataFrame({'获救':survived_1_three,'未获救':survived_0_three})
# df5.plot(kind='bar',stacked=True)
# plt.title('三等舱')
# plt.xlabel('性别')
# plt.ylabel('人数')


# 查看各船舱级别下的男女获救情况(方法二)
# 低等舱：Pclass=3；高等舱：Pclass=1/2
# 定义四个subplot：141,142,143,144
fig = figure()
plt.subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar',                                                                                                        color='pink')
plt.xticks(arange(2), ['未获救', '获救'])
plt.yticks(arange(0, 301, 50))
plt.legend(['低等舱/女生'], loc='best')

plt.subplot(142)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar',                                                                                                         color='hotpink')
plt.xticks(arange(2), ['未获救', '获救'])
plt.yticks(arange(0, 301, 50))
plt.legend(['高等舱/女生'], loc='best')

plt.subplot(143)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar',color='lightsteelblue')
plt.xticks(arange(2), ['未获救', '获救'])
plt.yticks(arange(0, 301, 50))
plt.legend(['低等舱/男生'], loc='best')

plt.subplot(144)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar',
                                                                                              color='cornflowerblue')
plt.xticks(arange(2), ['未获救', '获救'])
plt.yticks(arange(0, 301, 50))
plt.legend(['高等舱/男生'], loc='best')

fig.tight_layout()
plt.show()

'''
从性别上看，女性获救的机会比男性大
从船舱等级上看，三等舱获救机率较低
'''

fig = figure()
fig.set(alpha=0.2)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data_train = pd.read_csv("./data/train.csv")

plt.subplot(341)
data_train.Survived.value_counts().plot(kind='bar')
plt.title("获救情况（1为获救）")
plt.ylabel("人数")

plt.subplot(342)
data_train.Pclass.value_counts().plot(kind='bar')
plt.title("乘客等级分布")
plt.ylabel("人数")

plt.subplot(344)
y = data_train.Age
x = data_train.Survived
plt.scatter(x, y)
plt.title("按年龄看获救分布（1）为获救")
plt.ylabel("年龄")

plt.subplot(312)
y = data_train.Age.value_counts()
y.sort_index().plot(kind='bar')
plt.xlabel("年龄")
plt.ylabel("人数")
plt.title("各年龄的乘客人数")

plt.subplot(313)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')  # 密度图
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("年龄")
plt.ylabel("密度")
plt.title("各等级的乘客年龄分布")
plt.legend(('头等舱', '二等舱', '三等舱'), loc='best')

plt.subplot(343)
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("各登船口岸上船人数")
plt.ylabel("人数")
# subplots_adjust(wspace=0.5,hspace=1)#也可以调节各图形之间的间距
fig.tight_layout()  # 自动调整各图形之间的间距
plt.show()

'''
获救的几率与年龄有关
高等仓的平均年龄大于低等舱
'''