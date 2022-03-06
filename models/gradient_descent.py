from collections import defaultdict, deque
from enum import auto
from nbformat import current_nbformat  # Use this for effective implementation of L-BFGS
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time
import scipy

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

class DynamicRegressionOracle():
    def __init__(self, num_steps=5, learning_steps='all', autoregression_depth=5, fit_intercept=True):
        self.num_steps = num_steps
        if learning_steps == 'all':
            self.learning_steps = np.arange(num_steps)
        else:
            self.learning_steps = learning_steps
        self.autoregression_depth = autoregression_depth
        self.intercept_weight = int(fit_intercept)

    def calc_predicts(self, params, object):
        predicted_values = np.array([])
        for step in range(self.num_steps):
            func = params[0] * self.intercept_weight
            if step != 0:
                func += np.sum(predicted_values[-self.autoregression_depth:] * params[step:0:-1])
            if step < self.autoregression_depth:
                func += np.sum(
                    object[-(self.autoregression_depth - step):] * \
                    params[self.autoregression_depth:step:-1]
                )
            predicted_values = np.append(predicted_values, func)
        return predicted_values

    def calc_grad(self, params, object, true_targets):
        predicts = self.calc_predicts(params, object)
        grads = np.append([self.intercept_weight], object[::-1]).reshape(1, -1)
        for step in range(1, self.num_steps):
            object = np.append(object, predicts[step])
            object_part = np.append(
                [self.intercept_weight], 
                object[len(object) - 1:len(object) - self.autoregression_depth - 1:-1]
            )
            grad_part = np.sum(grads[-self.autoregression_depth:, :] * params[step:0:-1, np.newaxis], axis=0)
            grads = np.vstack([grads, object_part + grad_part])
        predicts = predicts[self.learning_steps]
        true_targets = true_targets[self.learning_steps]
        grads = grads[self.learning_steps]
        return np.average(grads * (2 * (predicts - true_targets))[:, np.newaxis], axis=0)

    def func(self, params, y):
        obj_indices = np.arange(len(y) - self.autoregression_depth - self.num_steps).reshape(-1, 1)
        error_func = np.vectorize(lambda i : mean_squared_error(
            self.calc_predicts(params, y[i:i + self.autoregression_depth]),
            y[i + self.autoregression_depth:i + self.autoregression_depth + self.num_steps]
        ), signature='()->()')
        errors = error_func(obj_indices).reshape(len(obj_indices))
        return np.average(errors)

    def grad(self, params, y):
        obj_indices = np.arange(len(y) - self.autoregression_depth - self.num_steps).reshape(-1, 1)
        grad_func = np.vectorize(lambda i : self.calc_grad(
            params,
            y[i:i + self.autoregression_depth],
            pd.Series(
                y[i + self.autoregression_depth:i + self.autoregression_depth + self.num_steps].values,
                index=range(self.num_steps)
            )
        ), signature='()->(n)')
        grads = grad_func(obj_indices).reshape(len(obj_indices), len(params))
        return np.average(grads, axis=0)

class GradientDescentOptimizer():
    def __init__(self, oracle, autoregression_depth=5, max_iter=100000, tolerance=1e-3, fit_intercept=True):
        self.oracle = oracle
        self.autoregression_depth = autoregression_depth
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.intercept_weight = int(fit_intercept)

    def optimize(self, y):
        intercept = 0
        weights = np.random.rand(self.autoregression_depth)
        weights /= np.sum(weights)
        x_0 = np.append([intercept], weights)
        x_k = np.copy(x_0)
        starting_grad = self.oracle.grad(x_0, y)
        starting_grad_norm = np.linalg.norm(starting_grad)
        current_grad = starting_grad  
        current_grad_norm = starting_grad_norm
        alpha = 1e-8
        for _ in range(self.max_iter):
            if current_grad_norm ** 2 <= self.tolerance * starting_grad_norm ** 2:
                break
            d = -current_grad
            # alpha = LearningRate()()
            x_k = x_k + alpha * d
            current_grad = self.oracle.grad(x_k, y)
            current_grad_norm = np.linalg.norm(current_grad)
        grad = self.oracle.grad(x_k, y)
        if np.square(np.linalg.norm(grad)) <= self.tolerance * starting_grad_norm ** 2:
            return x_k
        return x_k
