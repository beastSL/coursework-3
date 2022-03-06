from lib2to3.pytree import Base
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from dataclasses import dataclass
from gradient_descent import GradientDescentOptimizer, DynamicRegressionOracle

@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p

class DynamicRegression(BaseEstimator, TransformerMixin):
    def __init__(self, num_steps=5, learning_steps='all', autoregression_depth=5, fit_intercept=True, max_iter=100000, tolerance=1e-3):
        self.autoregression_depth = autoregression_depth
        self.oracle = DynamicRegressionOracle(num_steps=num_steps, learning_steps=learning_steps, autoregression_depth=autoregression_depth, fit_intercept=fit_intercept)
        self.optimizer = GradientDescentOptimizer(self.oracle, autoregression_depth, max_iter, tolerance, fit_intercept)

    def fit(self, y):
        self.params = self.optimizer.optimize(y)        

    def predict(self, y):
        return self.oracle.calc_predicts(
            self.params, 
            pd.Series(y[-self.autoregression_depth:].values, index=range(self.autoregression_depth))
        )