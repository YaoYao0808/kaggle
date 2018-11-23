'''
泰纳尼克号二分类问题预测—using stackingm
'''
import pandas as pd

df = pd.read_csv("../processed_data3.csv")
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# print("train_df",train_df)
# as_matrix()：Convert the frame to its Numpy-array representation.
train_np = train_df.as_matrix()
# print("train_np",train_np)
# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]


from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs


'''模型融合中使用到的各个单模型'''
clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

'''切分一部分数据作为测试集'''
X, X_predict, y, y_predict = train_test_split(X, y, test_size=0.33, random_state=2017)

print("X.shape",X.shape)
print("y.shape",y.shape)



dataset_blend_train = np.zeros((X.shape[0], len(clfs)))  #33500*5
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs))) #16500*5

df_test = pd.read_csv("../processed_test_data1.csv")

datatest = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

'''5折stacking'''
n_folds = 2
skf = StratifiedKFold(n_splits=n_folds)
# StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
kflods=list(skf.split(X,y))
# skf.get_n_splits(X, y)
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    print(j, clf)
    # 0 RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #     max_depth=None, max_features='auto', max_leaf_nodes=None,
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(kflods)))
    for i, (train, test) in enumerate(kflods):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        print("train",train,"test",test)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:,i] = clf.predict(datatest[1:296])
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
# clf = LogisticRegression()
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_blend_train, y)
print(dataset_blend_test.shape) #(295, 5)
y_submission = clf.predict(dataset_blend_test)

print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
#
# temp = pd.DataFrame(y_submission)
#
# y_submission.to_csv("ensemble_model.csv", index = False)

print(y_submission)

# result = pd.DataFrame( {'Survived': predictions.astype(np.int32)})


# y_submission.to_csv('ensemble_stacking_model.csv', index=False)
print("blend result")
print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))

# ****************************************
# 此模型已经训练好


# df_test = pd.read_csv("../processed_test_data1.csv")
#
# test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# print("test.shape",test.shape)  #(418, 2)
# predictions = clf.predict(test)
# result = pd.DataFrame( {'Survived': predictions.astype(np.int32)})
# result.to_csv("ensemble_predictions.csv", index=False)