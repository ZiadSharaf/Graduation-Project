from _read import *
from modules.train_test import brier_score_loss

from sklearn.metrics import confusion_matrix
import pickle

# read ensemble 1
with open("../models/ensemble_selection.pkl", 'rb') as f:
    ensemble_selection = pickle.load(f)

SELECTION = ['wc', 'fm', 'tg', 'sex', 'hdl', 'ggt', 'height', 'ast', 'hba1c', 'fpg', 'tc', 'sbp', 'age']

# calculate Brier score for each class
y_pred = ensemble_selection.predict_proba(X_test[SELECTION])
print(brier_score_loss(y_test[y_test==1], y_pred[:,1][y_test==1]))
print(brier_score_loss(y_test[y_test==0], y_pred[:,1][y_test==0]))
# construct confusion matrix
print(confusion_matrix(y_test, ensemble_selection.predict(X_test[SELECTION])))

# read ensemble 2
with open("../models/ensemble_nmf.pkl", 'rb') as f:
    ensemble_nmf = pickle.load(f)

# calculate Brier score for each class
y_pred = ensemble_nmf.predict_proba(X_test)
print(brier_score_loss(y_test[y_test==1], y_pred[:,1][y_test==1]))
print(brier_score_loss(y_test[y_test==0], y_pred[:,1][y_test==0]))
# construct confusion matrix
print(confusion_matrix(y_test, ensemble_nmf.predict(X_test)))

# read ensemble 3
with open("../models/ensemble_famd.pkl", 'rb') as f:
    ensemble_famd = pickle.load(f)

# converting nominal features to integer type to be treated as categorical
X_test[CATS] = X_test[CATS].astype("int")
# converting numerical features to integer type to be treated as numerical
X_test[NUMS] = X_test[NUMS].astype("float")
# converting ordinal features to integer type to be treated as numerical
X_test[ORDS] = X_test[ORDS].astype("float")

# calculate Brier score for each class
y_pred = ensemble_famd.predict_proba(X_test)
print(brier_score_loss(y_test[y_test==1], y_pred[:,1][y_test==1]))
print(brier_score_loss(y_test[y_test==0], y_pred[:,1][y_test==0]))
# construct confusion matrix
print(confusion_matrix(y_test, ensemble_famd.predict(X_test)))
