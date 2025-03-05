from modules.train_test import *
import modules.selection as selection
from _read import *
from modules.train_test import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import pickle

with open("../models/ensemble_scaled.pkl", 'rb') as f:
    ensemble_scaled = pickle.load(f)

# # get features importance
importances = selection.get_permutation_importances(ensemble_scaled, X_train, y_train)

# # plot feature importances
selection.plot(X_train.columns, importances)

# prompt to get number of features desired
n = int(input("Enter number of features to select: "))

sorted_cols = selection.get_sorted_cols(X_train.columns, importances)

SELECTION = sorted_cols[:n]

X_selection_train = X_train.loc[:,SELECTION]
X_selection_test = X_test.loc[:,SELECTION]

# ------------------------------------------------------------------------------------------

NUMS_SELECTION = [col for col in NUMS if col in SELECTION]
ORDS_SELECTION = [col for col in ORDS if col in SELECTION]

# hyperparameter tuning and training
ensemble_selection = tune_train(
    X_selection_train,
    y_train,
    X_selection_test,
    y_test,
    ('scaler', ColumnTransformer([('scaler', RobustScaler(), NUMS_SELECTION+ORDS_SELECTION)], remainder='passthrough', verbose_feature_names_out=False))
)

# saving ensemble
with open("../models/ensemble_selection.pkl", 'wb') as f:
    pickle.dump(ensemble_selection, f)
