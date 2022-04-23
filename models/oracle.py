from typing import List, Union
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

class DynamicRegressionOracle:
    def __init__(self,
                 fh: Union[int, List[int]]=1,
                 sp: Union[int, None]=None,
                 learning_steps: Union[str, List[int]]='all',
                 ar_depth: int=5,
                 seas_depth: Union[int,None]=2,
                 fit_intercept: bool=True):
        """
        Inner class that is used for calculating formulas for Dynamic Regression.

        Parameters
        ----------
        fh : Union[int, List[int]]=5
            Forecasting horizon for dynamic regression.
        sp : Union[int, None]=None
            Seasonal period. Includes seasonality in model.
        learning_steps : Union[str, List[int]]='all'
            Defines which steps of dynamic regression will be used for learning. Can be a list of steps or a string 'all'.
        ar_depth : int=5
            Defines how many lags will be included in the formula.
        seas_depth : Union[int,None]=2
            Defines how many seasonal lags will be included in the formula.
        fit_intercept : bool=True
            Whether to fit constant intercept or not.
        """
        self.fh = fh
        self.num_steps = np.max(fh)
        if learning_steps == 'all':
            self.learning_steps = np.arange(self.num_steps)
        else:
            self.learning_steps = learning_steps
        self.ar_depth = ar_depth
        self.intercept_weight = int(fit_intercept)
        self.sp = sp
        self.seas_depth = seas_depth
        self.max_depth = ar_depth
        self.ar_indices = -np.arange(1, ar_depth + 1)
        self.relative_indices = self.ar_indices
        if self.sp is not None:
            self.max_depth = np.max((ar_depth, sp * seas_depth))
            self.seas_indices = np.sort(np.setdiff1d(-np.arange(sp, sp * seas_depth + 1, sp), self.ar_indices))[::-1]
            self.relative_indices = np.sort(np.union1d(self.relative_indices, self.seas_indices))[::-1]

    def calc_predicts(self, params: List[float], object: List[float], fh: Union[int, List[int]]=None):
        """
        Predict next values of time series using Dynamic Regression.

        Parameters
        ----------
        params: List[float]
            Weights of Dynamic Regression.
        object: List[float]
            Previous samples of time series.
        fh: Union[int, List[int]]
            Forecasting horizon for prediction. Maximum value of fh must not exceed maximum value of fh that was given to constructor.

        Returns
        ----------
        predicted_values: List[float]
            Predicted values with indices that are specified in fh. 
        """
        if fh is None:
            fh = self.fh
        copy_object = np.copy(object)
        predicted_values = np.array([])
        for step in range(self.num_steps):
            intercept_part = params[0] * self.intercept_weight
            ar_indices = self.ar_indices[-self.ar_indices <= len(copy_object)]
            ar_part = np.sum(copy_object[ar_indices] * params[1:len(ar_indices) + 1])
            cur_predict = intercept_part + ar_part
            if self.sp is not None:
                seas_indices = self.seas_indices[-self.seas_indices <= len(copy_object)]
                seas_part = np.sum(copy_object[seas_indices] * params[self.ar_depth + 1:self.ar_depth + 1 + len(seas_indices)])
                cur_predict += seas_part
            predicted_values = np.append(predicted_values, cur_predict)
            copy_object = np.append(copy_object, cur_predict)
        return predicted_values[fh - 1]

    def calc_grad_for_one_object(self, params: List[float], object: List[float], true_targets: List[float]):
        """
        Calculate gradient of L2 loss for Dynamic Regression for one point of dataset.

        Parameters
        ----------
        params : List[float]
            Current weights of Dynamic Regression.
        object: List[float]
            Contiguous slice of time series that is used for prediction.
        true_targets: List[float]
            Slice of time series right after object that is used for calculating loss.

        Returns
        ----------
        grad: List[float]
            Gradient of L2 loss for Dynamic Regression for one point of dataset.
        """
        predicts = self.calc_predicts(params, object, np.arange(1, self.num_steps + 1))
        grads = np.append([self.intercept_weight], object[self.relative_indices]).reshape(1, -1)
        for step in range(1, self.num_steps):
            object = np.append(object, predicts[step - 1])
            object_part = np.append([self.intercept_weight], object[self.relative_indices])
            ar_indices = self.ar_indices[-self.ar_indices <= step]
            ar_grad_part = np.sum(grads[ar_indices, :] * params[1:len(ar_indices) + 1, None], axis=0)
            cur_grad = object_part + ar_grad_part
            if self.sp is not None:
                seas_indices = self.seas_indices[-self.seas_indices <= step]
                seas_grad_part = np.sum(grads[seas_indices, :] * params[self.ar_depth + 1:self.ar_depth + 1 + len(seas_indices), None], axis=0)
                cur_grad += seas_grad_part
            grads = np.vstack([grads, cur_grad])
        predicts = predicts[self.learning_steps]
        true_targets = true_targets[self.learning_steps]
        grads = grads[self.learning_steps]
        return np.average(grads * np.array(2 * (predicts - true_targets))[:, None], axis=0)


    def func(self, params: List[float], y: List[float]):
        """
        Calculate L2 loss for Dynamic Regression. It is calculated over fh given to constructor, then averaged over all points in train.

        Parameters
        ----------
        params : List[float]
            Current weights of Dynamic Regression
        y: List[float]
            Previous samples of time series

        Returns
        ----------
        error: List[float]
            Gradient of L2 loss for Dynamic Regression.
        """
        obj_indices = np.arange(len(y) - self.max_depth - self.num_steps + 1).reshape(-1, 1)
        error_func = np.vectorize(lambda i : mean_squared_error(
            [self.calc_predicts(params, y[i:i + self.max_depth])],
            [(y[i + self.max_depth:i + self.max_depth + self.num_steps])[self.fh - 1]]
        ), signature='()->()')
        errors = error_func(obj_indices).reshape(len(obj_indices))
        return np.average(errors)

    def grad(self, params: List[float], y: List[float]):
        """
        Calculate gradient of L2 loss for Dynamic Regression.

        Parameters
        ----------
        params : List[float]
            Current weights of Dynamic Regression
        y: List[float]
            Previous samples of time series

        Returns
        ----------
        grad: List[float]
            Gradient of L2 loss for Dynamic Regression.
        """
        obj_indices = np.arange(len(y) - self.max_depth - self.num_steps + 1).reshape(-1, 1)
        grad_func = np.vectorize(lambda i : self.calc_grad_for_one_object(
            params,
            object=y[i:i + self.max_depth],
            true_targets=pd.Series(
                y[i + self.max_depth:i + self.max_depth + self.num_steps],
                index=range(self.num_steps)
            )
        ), signature='()->(n)')
        grads = grad_func(obj_indices).reshape(len(obj_indices), len(params))
        return np.average(grads, axis=0)
