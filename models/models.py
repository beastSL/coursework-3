from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.trend import TrendForecaster

class SktimeETS(AutoETS):
    '''
    AutoETS model from sktime.
    '''
    def __init__(self, *args, **kwargs):
        super(SktimeETS, self).__init__(*args, **kwargs)

class SktimeARIMA(AutoARIMA):
    '''
    AutoARIMA model from sktime.
    '''
    def __init__(self, *args, **kwargs):
        super(SktimeARIMA, self).__init__(*args, **kwargs)

class SktimeLinearRegression(TrendForecaster):
    '''
    TrendForecaster model from sktime.
    '''
    def __init__(self, *args, **kwargs):
        super(SktimeLinearRegression, self).__init__(*args, **kwargs)
