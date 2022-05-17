from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster


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

class SktimeProphet(Prophet):
    '''
    PROPHET model from sktime.
    '''
    def __init__(self, *args, **kwargs):
        super(SktimeProphet, self).__init__(*args, **kwargs)

class SktimeNaive(NaiveForecaster):
    '''
    Naive model from sktime.
    '''
    def __init__(self, *args, **kwargs):
        super(SktimeNaive, self).__init__(*args, **kwargs)
