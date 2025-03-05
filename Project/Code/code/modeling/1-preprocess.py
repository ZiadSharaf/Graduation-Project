from modules.splitting import *

import pandas as pd

df = pd.read_csv("datasets/raw.csv")

redundent_features = [
    'bmi',
    'weight',
    'ethanol',
    'lbm',
    'alt',
    'dbp'
]

# delete the features
for feature in redundent_features:
    del df[feature]

df['smoking'] -= 1
df['drinking'] -= 1

X = df.iloc[:,:-1]
y = df['nafld']

X_train, X_test, y_train, y_test = split(X, y, 0.2)

X_train.to_csv("datasets/X_train.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)
