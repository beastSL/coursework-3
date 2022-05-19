from typing import Dict, List, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from models.gradient_descent import GradientDescentOptimizer
from lib.dynamic_regression_oracle import DynamicRegressionOracle


class DynamicRegression(BaseEstimator, TransformerMixin):
    def __init__(self,
                 fh: Union[int, List[int]]=1,
                 sp: Union[int, None]=None,
                 learning_steps: Union[str, List[int]]='fh',
                 ar_depth: int=1,
                 seas_depth: Union[int,None]=1,
                 fit_intercept: bool=True,
                 max_iter: int=10,
                 tolerance: float=1e-2):
        '''
        Dynamic Regression. Performs prediction using formula
        alpha_1 * y_{t-1} + ... + alpha_m * y_{t-m} + alpha_{m+1} * y_{t - (1 + m // sp) * sp} + ... + alpha_{m+n} * y_{t - (n + m // sp) * sp}.
        Difference between simple linear regression is that Dynamic Regression learns using its own predictions, not true y values.
        
        Parameters
        ----------
        fh : Union[int, List[int]]=1
            Forecasting horizon for dynamic regression.
        sp : Union[int, None]=None
            Seasonal period. Includes seasonality in model.
        learning_steps : Union[str, List[int]]='all'
            Defines which steps of dynamic regression will be used for learning. Can be a list of steps or a string 'all'.
        ar_depth : int=5
            Defines how many lags will be included in the formula (it's the `m` parameter in the formula). 
        seas_depth : Union[int,None]=2
            Defines how many seasonal lags will be included in the formula (it's the `n + m // sp` parameter in the formula).
        fit_intercept : bool=True
            Whether to fit constant intercept or not.
        max_iter : int=10000
            Defines at how many iterations will the optimization stop.
        tolerance : float=1e-2
            Defines at which relative tolerance will the optimization stop.
        '''
        fh = self._to_numpy(fh)
        if learning_steps == 'fh':
            learning_steps = fh
        elif learning_steps == 'all':
            learning_steps = np.arange(np.max(fh))
        self.oracle = DynamicRegressionOracle(self._to_numpy(fh), sp, learning_steps, ar_depth, seas_depth, fit_intercept)
        self.optimizer = GradientDescentOptimizer(self.oracle,
                                                  ar_depth + int(np.ceil((sp * seas_depth - ar_depth) / sp) if sp is not None else 0),
                                                  fit_intercept,
                                                  max_iter,
                                                  tolerance,)

    def fit(self, y: Union[List[float], np.ndarray, pd.Series], trace: bool=False, lr_finder: str='Formula', display=False, starting_params: str='random') -> Tuple[str, Dict]:
        """
        Fit weights of Dynamic Regression.

        Parameters
        ----------
        y: Union[List[float], np.ndarray, pd.Series]
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
        status: str
            "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.

        history : dictionary of lists or None
            Dictionary containing the progress information or None if trace=False.
            Dictionary has to be organized as follows:
                - history['time'] : list of floats, containing time in seconds passed from the start of the method
                - history['func'] : list of function values f(x_k) on every step of the algorithm
                - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
        """
        self.params, status, history = self.optimizer.optimize(self._to_numpy(y), 
                                                               trace=trace,
                                                               lr_finder=lr_finder,
                                                               display=display,
                                                               starting_params=starting_params)
        return status, history

    def predict(self, y: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict next values of time series using Dynamic Regression.

        Parameters
        ----------
        y: Union[List[float], np.ndarray, pd.Series]
            Previous samples of time series.

        Returns
        ----------
        predicted_values: np.ndarray
            Predicted values of time series specified by fh.
        """
        return self.oracle.calc_predicts(self.params.astype(np.float64), self._to_numpy(y).astype(np.float64))

    def _to_numpy(self, y: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert object of type list, pandas.Series or numpy.ndarray to numpy.ndarray

        Parameters
        ----------
        y: Union[List[float], np.ndarray, pd.Series]
            Previous samples of time series or forecasting horizaon.

        Returns
        ----------
        y.to_numpy(): np.ndarray
            y converted to numpy.ndarray.
        """
        if isinstance(y, pd.Series):
            return y.to_numpy()
        elif isinstance(y, list) or isinstance(y, int):
            return np.array(y).flatten()
        elif isinstance(y, np.ndarray):
            return y
        else:
            raise TypeError("Only objects of type list, pandas.Series and numpy.ndarray are valid.")
