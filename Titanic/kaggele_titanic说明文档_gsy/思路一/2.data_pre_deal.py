# 2.数据预处理
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 设置结果显示完整
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data_train = pd.read_csv("./data/train.csv")

'''
发现Age(年龄)缺失的数据较少，所以采用插值法来处理。用到的插值方法是随机森林
而Cabin（船舱号）则缺失了约3/4，而船舱号码暂时看不出来对获救有什么影响
所以我选择了对其进行数值化，有船舱号的值记为为1，没有船舱号的记为0
同样，为了方便后面的建模，对非数值的属性Sex（性别），Embarked（登舱口），Name（名字），Ticket（票号）进行同样的数值化处理。这里用到了pandas库中的dummies函数。
'''

# 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    print("into set_missing_ages")
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print("know_age:",known_age)
    # print("unknow_age",unknown_age)

    y = known_age[:, 0]  # 不为空的第一列所有元素
    x = known_age[:, 1:]  # 分割出不为空矩阵第二列以后的所有元素
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)
    predictedAges = rfr.predict(unknown_age[:, 1:])
    # print("predictedAges:",predictedAges)
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges  # loc通过行标签来索引数据
    return df


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
# print("data_train",data_train.head(30))
data_train.to_csv('processed_data1.csv')  # 新文件夹没有空值

# 对非数值的属性Sex（性别），Embarked（登舱口），Name（名字），Ticket（票号）进行数值化处理
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass],
               axis=1)  # 增加列属性,concat默认是增加行，axis=1为增加列
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)  # drop为删除原来的列，axis=1为删除列
df.to_csv('processed_data2.csv')  # 新文件夹的数据都为数值型,对原数据进行了特征因子化

#将Age和Fare两个属性进行零——均值标准化。
scale_age=(df['Age']-df['Age'].mean())/df['Age'].std()
scale_fare=(df['Fare']-df['Fare'].mean())/df['Fare'].std()
df.copy()
df['Age']=scale_age
df['Fare']=scale_fare
df.to_csv('processed_data3.csv')

