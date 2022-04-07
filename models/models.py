from sklearn.base import BaseEstimator, TransformerMixin
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.trend import TrendForecaster

class SktimeETS(AutoETS):
    def __init__(self, *args, **kwargs):
        super(SktimeETS, self).__init__(*args, **kwargs)

class SktimeARIMA(AutoARIMA):
    def __init__(self, *args, **kwargs):
        super(SktimeARIMA, self).__init__(*args, **kwargs)

class SktimeLinearRegression(TrendForecaster):
    def __init__(self, *args, **kwargs):
        super(SktimeLinearRegression, self).__init__(*args, **kwargs)
