from modules.constants import *
from modules.constants_tuning import *
from modules.clfs import *

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from skopt import BayesSearchCV
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.metrics import brier_score_loss

def make_pipe(clf_name, clf, preprocessing_step):
    if (clf_name == 'knn' or clf_name == 'svm'):
        # pipeline steps: undersampling -> preprocessing -> classifier
        return Pipeline([('sampler', RandomUnderSampler()), preprocessing_step, ('clf', clf)])

    elif (clf_name == 'rf' or clf_name == 'xgb'):
        # if the preprocessing step is scaling
        if preprocessing_step[0] == 'scaler':
            # pipeline steps: classifier
            return Pipeline([('clf', clf)])
        # if the preprocessing step is not scaling
        else:
            # pipeline steps: preprocessing -> classifier
            return Pipeline([preprocessing_step, ('clf', clf)])

    elif (clf_name == 'bn'):
        # if the preprocessing step is scaling
        if preprocessing_step[0] == 'scaler':
            # pipeline steps: undersampling -> classifier
            return Pipeline([('sampler', RandomUnderSampler()), ('clf', clf)])

        # if the preprocessing step is not scaling
        else:
            # pipeline steps: undersampling -> preprocessing -> classifier
            return Pipeline([('sampler', RandomUnderSampler()), preprocessing_step, ('clf', clf)])

def tune_train(X_train, y_train, X_test, y_test, preprocessing_step):
    preprocessing_step[1].set_output(transform='pandas')

    clf_pipes = []
    for clf_name, clf in clfs_dict.items():
        clf_pipe = make_pipe(clf_name, clf, preprocessing_step)
        optimizer = BayesSearchCV(
            clf_pipe,
            search_spaces[clf_name],
            # perform startified 5-fold cross-validation
            cv=5,
            n_iter=n_iters[clf_name],
            scoring=custom_brier,
            refit=False,
            n_jobs=NJOBS
        )
        optimizer.fit(X_train, y_train)

        # set hyperparameters to learned ones
        clf_pipe.set_params(**optimizer.best_params_)

        # create a new instance of FAMD because it fails to refit
        if preprocessing_step[0] == 'famd':
            from prince import FAMD
            famd = FAMD(preprocessing_step[1].n_components)
            clf_pipe.steps[-2] = ('famd', famd)

        clf_pipe.fit(X_train, y_train)

        clf_pipes.append(clf_pipe)

    # calculate scores
    scores = [custom_brier(clf_pipe, X_test, y_test) for clf_pipe in clf_pipes]

    # combining models into an ensemble
    weights = [score/sum(scores) for score in scores]
    ensemble = EnsembleVoteClassifier(
        clf_pipes,
        voting="soft",
        weights=weights,
        fit_base_estimators=False,
        use_clones=False
    ).fit(X_train, y_train)

    return ensemble

def custom_brier(clf, X, y):
    y_pred = clf.predict_proba(X)

    # calculate brier score for positive samples only
    brier_pos = brier_score_loss(y[y==1], y_pred[:,1][y==1])

    # calculate brier score for negative samples only
    brier_neg = brier_score_loss(y[y==0], y_pred[:,1][y==0])

    return 1 - (brier_pos + brier_neg)/2
