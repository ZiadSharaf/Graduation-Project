from modules.train_test import *
from modules.selection import *

from _read import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import NMF
from prince import FAMD
import pickle

# hyperparameter tuning and training
ensemble_scaled = tune_train(
    X_train,
    y_train,
    X_test,
    y_test,
    ('scaler', ColumnTransformer([('scaler', RobustScaler(), NUMS+ORDS)], remainder='passthrough', verbose_feature_names_out=False))
)

# saving ensemble
with open("../models/ensemble_scaled.pkl", 'wb') as f:
    pickle.dump(ensemble_scaled, f)

# ------------------------------------------------------------------------------------------

# NMF object
nmf = NMF(7, init='nndsvd', max_iter=100000)

# hyperparameter tuning and training
ensemble_nmf = tune_train(
    X_train,
    y_train,
    X_test,
    y_test,
    ('nmf', ColumnTransformer([('nmf', nmf, NUMS)], remainder='passthrough', verbose_feature_names_out=False))
)

# saving ensemble
with open("../models/ensemble_nmf.pkl", 'wb') as f:
    pickle.dump(ensemble_nmf, f)

# ------------------------------------------------------------------------------------------

# converting nominal features to integer type to be treated as categorical
X_train[NOMS] = X_train[NOMS].astype("int")
X_test[CATS] = X_test[CATS].astype("int")
# converting numerical features to integer type to be treated as numerical
X_train[NUMS] = X_train[NUMS].astype("float")
X_test[NUMS] = X_test[NUMS].astype("float")
# converting ordinal features to integer type to be treated as numerical
X_train[ORDS] = X_train[ORDS].astype("float")
X_test[ORDS] = X_test[ORDS].astype("float")
# FAMD object
famd = FAMD(11)

# hyperparameter tuning and training
ensemble_famd = tune_train(X_train, y_train, X_test, y_test, ('famd', famd))

# saving ensemble
with open("../models/ensemble_famd.pkl", 'wb') as f:
    pickle.dump(ensemble_famd, f)
