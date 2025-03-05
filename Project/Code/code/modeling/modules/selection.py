from sklearn.inspection import permutation_importance
from modules.constants import NJOBS
from modules.train_test import custom_brier
import matplotlib.pyplot as plt

def get_permutation_importances(est, some_X, some_y):
    results = permutation_importance(
        est,
        some_X,
        some_y,
        n_jobs=NJOBS,
        scoring=custom_brier
    )
    # get the average importance across all 5 repititions
    return results['importances_mean']

def get_sorted_cols(cols, importances):
    return [index for _, index in sorted(zip(importances, cols), reverse=True)]

def plot(cols, importances):
    sorted_cols = get_sorted_cols(cols, importances)
    plt.barh(sorted_cols, sorted(importances, reverse=True))
    plt.title("Features Importances")
    plt.show()
