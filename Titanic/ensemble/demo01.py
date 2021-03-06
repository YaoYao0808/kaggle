'''
method:StratifiedKFold
'''
import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)  #Number of folds. Must be at least 2.
skf.get_n_splits(X, y)  #Returns the number of splitting iterations in the cross-validator

print(skf)
#StratifiedKFold(n_splits=2, random_state=None, shuffle=False)

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
'''
TRAIN: [1 3] TEST: [0 2]
TRAIN: [0 2] TEST: [1 3]
'''

# **************************************************************************************
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)
# KFold(n_splits=2, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
'''
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]
'''



