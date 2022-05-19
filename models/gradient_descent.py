from functools import partial, partialmethod
from typing import Dict, List, Union, Tuple
from collections import defaultdict
import time
from statsmodels.tsa.stattools import pacf

import numpy as np
from scipy.optimize import line_search
from lib.dynamic_regression_oracle import DynamicRegressionOracle
from sklearn.exceptions import ConvergenceWarning
import warnings

class DynamicRegressionOracleWrapper:
    def __init__(self, oracle: DynamicRegressionOracle, y: np.ndarray):
        self.y = y
        self.oracle = oracle
    
    def func(self, params: np.ndarray) -> float:
        return self.oracle.func(params.astype(np.float64), self.y.astype(np.float64))

    def grad(self, params: np.ndarray) -> np.ndarray:
        return self.oracle.grad(params.astype(np.float64), self.y.astype(np.float64))


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Formula' -- choose lr using formula alpha_k = lambda * (s_0 / (s_0 + k))^p
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
        If method == 'Formula':
            lambda, s_0, p : Parameters for formula
    """
    def __init__(self, method, **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1e-3)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1e-3)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1e-3)
        elif self._method == 'Formula':
            self.lam = kwargs.get('c1', 5e-3)
            self.s_0 = kwargs.get('c2', 1)
            self.p = kwargs.get('alpha_0', 0.5)
            self.k = 0
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, y, x, d, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        func: function
            Returns L2 loss for dynamic regression
        grad: function
            Returns gradient L2 loss for dynamic regression
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x : np.array
            Starting point
        d : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == "Constant":
            return self.c
        if self._method == 'Formula':
            self.k += 1
            return self.lam * (self.s_0 / (self.s_0 + self.k)) ** self.p
        wrapper = DynamicRegressionOracleWrapper(oracle, y)
        func = wrapper.func
        grad = wrapper.grad
        if self._method == 'Wolfe':
            alpha = line_search(func, grad, x, d, c1=self.c1, c2=self.c2)[0]
            if alpha is not None:
                return alpha
        alpha = self.alpha_0
        if previous_alpha is not None:
            alpha = previous_alpha * 2
        while np.squeeze(func(x + alpha * d)) > \
              func(x) + self.c1 * alpha * np.squeeze(grad(x + alpha * d).dot(d)):
            alpha /= 2
        return alpha

class GradientDescentOptimizer:
    def __init__(self,
                 oracle: DynamicRegressionOracle,
                 num_weights: int,
                 fit_intercept: bool=True,
                 max_iter: int=10000,
                 tolerance: float=1e-2):
        """
        Class for Dynamic Regression weights optimization using gradient descent.

        Parameters
        ----------
        oracle: DynamicRegressionOracle
            An instance of DynamicRegressionOracle, defined from the same DynamicRegression instance as this class.
        num_weights: int
            Number of weights to optimize (excluding intercept)
        fit_intercept : bool=True
            Whether to fit constant intercept or not.
        max_iter : int=10000
            Defines at how many iterations will the optimization stop.
        tolerance : float=1e-2
            Defines at which relative tolerance will the optimization stop.
        """
        self.oracle = oracle
        self.num_weights = num_weights
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.intercept_weight = int(fit_intercept)

    def optimize(self, y: List[float], trace: bool=False, lr_finder: str='Wolfe', display=False, starting_params: str='random') -> Tuple[List[float], Union[Dict, None], str]:
        """
        Optimize weights of Dynamic Regression.

        Parameters
        ----------
        y : List[float]
            Previous samples of time series.
        trace: bool=False
            Whether to return history or not.
        lr_finder: str='Wolfe'
            Which method to use for lr choosing.
        display: bool=False
            Whether to print debug information or not.
        starting_params: str='random'
            How to initialize starting parameters for dynamic regression. Accepts the following values:
            'random' (default): initialize them randomly and normalize (so that the sum of params equals to 1)
            'pacf': initialize as partial autocorrelations of the series
            'pacf_norm': initialize as partial autocorrelations of the series and normalize

        Returns
        ----------
        weights: List[float]
            Optimized weights.
        status: str
            "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
        history : dictionary of lists or None
            Dictionary containing the progress information or None if trace=False. Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
        """
        history = defaultdict(list) if trace else None
        intercept = 0
        if starting_params == 'random':
            weights = np.random.rand(self.num_weights)
            weights /= np.sum(weights)
        elif starting_params == 'pacf':
            weights = pacf(y, nlags=self.num_weights)[1:]
        elif starting_params == 'pacf_norm':
            weights = pacf(y, nlags=self.num_weights)[1:]
            weights /= np.sum(weights)
        else:
            raise ValueError()
        x_0 = np.append([intercept], weights)
        x_k = np.copy(x_0)
        # print(x_0, y)
        starting_grad = self.oracle.grad(x_0.astype(np.float64), y.astype(np.float64)) 
        # print(starting_grad)
        starting_grad_norm = np.linalg.norm(starting_grad)
        current_grad = starting_grad  
        current_grad_norm = starting_grad_norm
        alpha = None
        line_search_tool = LineSearchTool(method=lr_finder)
        start = time.time()
        if trace:
            history['time'].append(0)
            history['func'].append(self.oracle.func(x_k.astype(np.float64), y.astype(np.float64)))
            history['grad_norm'].append(current_grad_norm)
        for _ in range(self.max_iter):
            if current_grad_norm ** 2 <= self.tolerance * starting_grad_norm ** 2:
                break
            d = -current_grad / np.linalg.norm(current_grad)
            alpha = line_search_tool.line_search(self.oracle, y, x_k, d, alpha)
            if display:
                print(f"Iter {_}")
                print(f"Params {x_k}")
                print(f"Starting grad norm {starting_grad_norm}")
                print(f"Current grad norm {current_grad_norm}")
                print(f"Alpha {alpha}")
                print(f"Func {self.oracle.func(x_k.astype(np.float64), y.astype(np.float64))}")
                print(f"Tolerance {current_grad_norm ** 2 / starting_grad_norm ** 2}")
            new_x_k = x_k + alpha * d
            if np.any(np.isnan(d) | np.isinf(d) | (d == None)) or \
               np.isnan(alpha) or np.isinf(alpha) or alpha == None or \
               np.any(np.isnan(new_x_k) | np.isinf(new_x_k) | (new_x_k == None)):
                return x_k, 'computational_error', history
            x_k = x_k + alpha * d
            # print(x_k)
            old_grad_norm = current_grad_norm
            current_grad = self.oracle.grad(x_k.astype(np.float64), y.astype(np.float64))
            current_grad_norm = np.linalg.norm(current_grad)
            if old_grad_norm == current_grad_norm:
                break
            new_iter = time.time()
            if trace:
                history['time'].append(new_iter - start)
                history['func'].append(self.oracle.func(x_k.astype(np.float64), y.astype(np.float64)))
                history['grad_norm'].append(current_grad_norm)
        if current_grad_norm ** 2 > self.tolerance * starting_grad_norm ** 2:
            warnings.warn("Gradient descent did not converge", ConvergenceWarning)
            return x_k, 'iterations_exceeded', history
        return x_k, 'success', history
