# 逻辑回归做拟合
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import preprocessing
# from 2.data_pre_deal import set_Cabin_type

# 设置结果显示完整
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


df = pd.read_csv("processed_data3.csv")

'''
processed_data3.csv:PassengerId Survived	Age 	SibSp	Parch	Fare	
                    Cabin_No	Cabin_Yes	Embarked_C	Embarked_Q	Embarked_S	
                    Sex_female	Sex_male	Pclass_1	Pclass_2	Pclass_3
train_df:           Survived  SibSp  Parch  Cabin_No  Cabin_Yes  Embarked_C  Embarked_Q  Embarked_S  
                    Sex_female  Sex_male  Pclass_1  Pclass_2  Pclass_3
用正则过滤从processed_data3.csv中取出train_df数据
'''

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# print("train_df",train_df)
# as_matrix()：Convert the frame to its Numpy-array representation.
train_np = train_df.as_matrix()
# print("train_np",train_np)
# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到LogisticRegression之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

####对测试数据同样进行数据预处理
data_test = pd.read_csv("./data/test.csv")
data_train = pd.read_csv("./data/train.csv")
# 构建同一个随机森林
age_df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]  # 第一列所有元素
x = known_age[:, 1:]  # 分割出矩阵第二列以后的所有元素
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(x, y)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test.to_csv('processed_test_data1.csv')  # 新文件夹没有空值

# 将Age和Fare两个属性进行零——均值标准化。
scale_age = (df['Age'] - df['Age'].mean()) / df['Age'].std()
scale_fare = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()
df.copy()
df['Age'] = scale_age
df['Fare'] = scale_fare
df.to_csv('processed_test_data2.csv')

# 用逻辑回归模型做预测
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)


y_pred=predictions
y_true=pd.read_csv("./data/gender_submission.csv").Survived

# 计算准确率
from sklearn.metrics import accuracy_score
# 评价指标 精度=（TP + TN）/（TP + TN + FP + FN）
print("accuracy_score:",accuracy_score(y_true, y_pred))
