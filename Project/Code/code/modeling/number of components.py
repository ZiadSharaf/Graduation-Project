from modules.constants import *
from _read import *

from sklearn.decomposition import NMF
from prince import FAMD

# perform nmf for a number of components from 1 to 11
for n in range(1, 12):
	nmf = NMF(n, init='nndsvd', max_iter=100000)
	nmf.fit_transform(X_train, y_train)
	# print reconstruction error
	print(nmf.reconstruction_err_)

# converting nominal features to integer type to be treated as categorical
X_train[CATS] = X_train[CATS].astype("int")
# converting numerical features to integer type to be treated as numerical
X_train[NUMS] = X_train[NUMS].astype("float")
# converting ordinal features to integer type to be treated as numerical
X_train[ORDS] = X_train[ORDS].astype("float")
# perform famd for a number of components from 1 to 11
for n in range(1, 12):
	famd = FAMD(n)
	famd.fit_transform(X_train, y_train)
	# print total percentage of explained variance
	print(famd.cumulative_percentage_of_variance_[-1])
