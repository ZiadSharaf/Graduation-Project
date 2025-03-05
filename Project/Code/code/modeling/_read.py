from modules.constants import *

import pandas as pd

# reading the data

X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv")['nafld']
y_test = pd.read_csv("datasets/y_test.csv")['nafld']

# converting categorical features to datatypes expected by scikit-learn

X_train[NOMS] = X_train[NOMS].astype(pd.CategoricalDtype())
X_test[NOMS] = X_test[NOMS].astype(pd.CategoricalDtype())
X_train[ORDS] = X_train[ORDS].astype(pd.CategoricalDtype(ordered=True))
X_test[ORDS] = X_test[ORDS].astype(pd.CategoricalDtype(ordered=True))
