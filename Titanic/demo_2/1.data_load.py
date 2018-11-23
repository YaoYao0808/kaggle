# NumPy
import numpy as np

# Dataframe operations
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# 设置结果显示完整
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 1.Loading datasets
'''
此处注意数据的处理方式：
将训练集和测试集合并成数据data_df，方便进行统一处理
但是在处理完成后，再将训练集和测试集按照891的长度分开
'''
train_df = pd.read_csv("../data/train.csv")   #891
test_df = pd.read_csv("../data/test.csv")     #418
data_df = train_df.append(test_df) # The entire data: train + test.  1309

# 2.特征处理
'''
age:
从Name属性中提取出姓氏Title,
按照Title分组，并使用各组Age的中位数替换该组Age的空值，
年龄的并未有很大的影响
'''
data_df['Title'] = data_df['Name']
# Cleaning name and extracting Title     从Name属性中提取出姓，并新增属性Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# print("Title:",data_df["Title"])

# Replacing rare titles with more common ones
# Mlle,Major,Col,Sir,Don等Title名都是很少出现，因此将其替换为常出现的title
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
           'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    print("age_to_impute:",age_to_impute)
    # 此处的age_to_impute为按Title分组后各组中Age的均值
    '''
        age_to_impute: 49.0
        age_to_impute: 4.0
        age_to_impute: 22.0
        age_to_impute: 30.0
        age_to_impute: 36.0
        age_to_impute: 41.5
    '''
    # 将各组中Age为空的值替换为该组的中位数
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

# Dropping Title feature
data_df.drop('Title', axis=1, inplace=True)
# print(data_df.head())

'''
增加家庭成员数量
Adding Family_Size,That's just Parch + SibSp.
'''
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]

'''
Adding Family_Survival
'''
# 获取名字属性，并将Fare中的空值用其均值填充
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

# 选择数值型属性进行遍历
# 以家庭和船票价格为分组，进行统计存活的数量
for grp, grp_df in data_df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    # print("grp:",grp)
    # print("grp_df",grp_df)
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            # 默认值0.5，若为该组中最大值，则将对应的乘客的Family_Survival赋值为1，否则赋值为0
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Family_Survival",data_df['Family_Survival'])
print("Number of passengers with family survival information:",
      data_df.loc[data_df['Family_Survival'] != 0.5].shape[0])

# 按照船的型号Ticket进行分组
for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5): #注意此处的判断条件
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passenger with family/group survival information: "
      + str(data_df[data_df['Family_Survival'] != 0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]

'''
Making FARE BINS
将船票价格Fare离散化，分为五个档次
'''
data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
# print("data_df['FareBin_Code']",data_df['FareBin_Code']

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)

'''
Making AGE BINS
将年龄离散化
注意：若要用中位数/均值等替代，最好是取整个数据样本
'''
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)

'''
Mapping SEX and cleaning data (dropping garbage)
'''
print(train_df.head(10))
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)


train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)

# 此时剩下属性：Survived  Pclass  Sex  Family_Size  Family_Survival  FareBin_Code  AgeBin_Code

# **********************************************************************************************************************
'''
训练模型
'''
# 创建X，y
X = train_df.drop('Survived', 1)
y = train_df['Survived']
X_test = test_df.copy()

# Scaling features
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

# Grid Search CV   KNN为例  网格搜索
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)  #0.879492358564122
print(gd.best_estimator_)

# Using a model found by grid searching
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

# 存入结果
result = pd.DataFrame({'Survived': y_pred.astype(np.int32)})
result.to_csv("knn_predictions.csv", index=False)

# 使用XGBoost  0.82296
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X,y)
result = pd.DataFrame({'Survived': y_pred.astype(np.int32)})
result.to_csv("xgboost_predictions.csv", index=False)




# Using a model found by grid searching
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2,
                           weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test)

temp = pd.DataFrame(pd.read_csv("../data/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("knn_predictions2.csv", index = False)
