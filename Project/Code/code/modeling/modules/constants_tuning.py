from skopt.space import Categorical, Integer, Real

search_spaces = {
    'knn': {'clf__n_neighbors': Integer(1, 200), 'clf__p': Integer(1, 4)},
    'svm': {'clf__C': Real(0.01, 100)},
    'rf': {
        'clf__max_leaf_nodes': Integer(2, 100),
        'clf__criterion': Categorical(['gini', 'entropy'])
    },
    'xgb': {
        'clf__max_leaves': Integer(2, 100),
        'clf__learning_rate': Real(0.01, 0.99)
    },
    'bn': [
        ({ # search space 1
            'clf__dag_structure': Categorical(['tan']),
            'clf__n_bins': Integer(1, 100)
        }, 25), # 25 iterations
        ({ # search space 2
            'clf__dag_structure': Categorical(['ban', 'gbn']),
            'clf__scoring_method': Categorical(
                ["k2score", "bdeuscore", "bdsscore", "bicscore", "aicscore"]
            ),
            'clf__n_bins': Integer(1, 100),
            'clf__max_indegree': Integer(2, 10)
        }, 100) # 100 iterations
    ]
}

# number of bayesian optimizaiton iterations
# unused for algorithms whose search spaces have specified number of iterations
n_iters = {
    'knn':  50,
    'svm':  25,
    'rf':   50,
    'xgb':  50,
    'bn':   None
}
