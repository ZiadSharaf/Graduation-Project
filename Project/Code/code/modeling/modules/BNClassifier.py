from modules.constants import *

import pandas as pd
from sklearn.base import ClassifierMixin
from pgmpy.estimators import HillClimbSearch, TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

class BNClassifier(BayesianNetwork, ClassifierMixin):

    def __init__(self, scoring_method='k2score', n_bins=6, dag_structure='ban', max_indegree=3):
        super().__init__()
        self.scoring_method = scoring_method
        self.n_bins = n_bins
        self.dag_structure = dag_structure
        self.max_indegree = max_indegree

    def fit(self, X, y):
        # ensure all columns names have same datatype (string)
        X = X.rename(columns={col: str(col) for col in X.columns})

        # convert NAFLD to a Pandas Series of categorical datatype
        y = pd.Series(y, name='nafld', dtype=pd.CategoricalDtype())

        df = pd.concat([X, y], axis=1)

        # constructing discretization bins
        self.bins = {}
        for col in df.columns:
            # if feature is not already categorical
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                self.bins[col] = pd.qcut(df[col], self.n_bins, retbins=True, duplicates='drop')[1]
                # set the the first bin to -inf and last bin to inf
                # to accept any value
                self.bins[col][0] = -float("inf")
                self.bins[col][-1] = float("inf")

        # discretization
        df = self.discretize(df)

        # structure learning
        self.learn_structure(df, X.columns)

        # parameter learning
        super().fit(df, estimator=BayesianEstimator)

        return self
    
    def learn_structure(self, df, features):
        if self.dag_structure == 'gbn':
            # learn the structure of a general bayesian network
            # through hill climbing
            learned_dag = HillClimbSearch(df).estimate(
                scoring_method=self.scoring_method,
                show_progress=False,
                max_indegree=self.max_indegree
            )

        elif self.dag_structure == 'ban':
            # learn the structure of a bayesian augmented naive bayes
            # through hill climbing with forced naive bayes edges
            learned_dag = HillClimbSearch(df).estimate(
                scoring_method=self.scoring_method,
                # force edges from NAFLD to all the features
                fixed_edges=[('nafld', col) for col in features],
                show_progress=False,
                max_indegree=self.max_indegree
            )

        elif self.dag_structure == 'tan':
            # learn the structure of a tree augmented naive bayes
            # through the chow-liu algorithm
            learned_dag = TreeSearch(df, n_jobs=1).estimate(
                estimator_type='tan',
                class_node='nafld'
            )

        # reset the network if already fitted
        while self.nodes():
            node = list(self.nodes)[0]
            self.remove_node(node)

        # add the learned edges
        self.add_edges_from(learned_dag.edges)

    def discretize(self, df):
        # ensure all columns names have same datatype (string)
        df = df.rename(columns={col: str(col) for col in df.columns})

        discretized_df = df.copy()
        for col in df.columns:
            if col in self.bins.keys():
                discretized_df[col] = pd.cut(df[col], self.bins[col])
        return discretized_df

    def predict(self, X):
        X = self.discretize(X)
        y_pred = super().predict(X).to_numpy()
        return y_pred

    def predict_proba(self, X):
        X = self.discretize(X)
        probs = super().predict_probability(X).to_numpy()
        return probs

    def predict_probability(self, df):
        df = self.discretize(df)
        return super().predict_probability(df)

    def get_params(self, deep=True):
        return {
            'scoring_method': self.scoring_method,
            'n_bins': self.n_bins,
            'dag_structure': self.dag_structure,
            'max_indegree': self.max_indegree
        }

    def set_params(self, **params):
        for param in params:
            setattr(self, param[0], param[1])
        return self
