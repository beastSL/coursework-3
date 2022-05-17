import pandas as pd
import numpy as np
from pprint import pprint
from sktime.forecasting.model_selection import temporal_train_test_split
from models.dynamic_regression import DynamicRegression
from models.models import SktimeETS, SktimeARIMA, SktimeProphet, SktimeNaive
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dataset = pd.read_csv("/Users/beast-sl/source/repos/coursework-3/Dataset/Train/Monthly-train.csv")
    testing_series = dataset[dataset["V1"] == "M100"].squeeze().dropna().drop(index='V1')
    testing_series.index = range(len(testing_series))
    testing_series = testing_series.astype(np.float).to_numpy()
    train, test = temporal_train_test_split(testing_series, test_size=5)

    print("Testing Dynamic Regression")
    model = DynamicRegression(fh=np.arange(1, 6))
    status, history = model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))), trace=True, display=True)
    print(f"Status {status}, history:{history}")
    pprint(history)
    dynamic_regression_predicts = model.predict(train)
    print(f"MSE: {mean_squared_error(dynamic_regression_predicts, test)}")

    print("Testing seasonal Dynamic Regression")
    model = DynamicRegression(fh=np.arange(1, 6), sp=12)
    status, history = model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))), trace=True, display=True)
    print(f"Status {status}, history:")
    pprint(history)
    seasonal_dynamic_regression_predicts = model.predict(train)
    print(f"MSE: {mean_squared_error(seasonal_dynamic_regression_predicts, test)}")

    print("Testing ETS")
    model = SktimeETS(auto=True, n_jobs=-1)
    model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))))
    ets_predicts = model.predict(fh=np.arange(1, 6))
    print(f"MSE: {mean_squared_error(ets_predicts, test)}")

    print("Testing ARIMA")
    model = SktimeARIMA(suppress_warnings=True)
    model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))))
    arima_predicts = model.predict(fh=np.arange(1, 6))
    print(f"MSE: {mean_squared_error(arima_predicts, test)}")

    print("Testing Prophet")
    model = SktimeProphet()
    model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))))
    prophet_predicts = model.predict(fh=np.arange(1, 6)).to_numpy()
    print(f"MSE: {mean_squared_error(prophet_predicts, test)}")

    print("Testing Naive")
    model = SktimeNaive()
    model.fit(pd.Series(train, index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq='M', periods=len(train)))))
    naive_predicts = model.predict(fh=np.arange(1, 6)).to_numpy()
    print(f"MSE: {mean_squared_error(naive_predicts, test)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(train) + len(test)), y=np.append(train, test), name="Test"))
    fig.add_trace(go.Scatter(x=np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], dynamic_regression_predicts),
                             name="Dynamic regression"))
    fig.add_trace(go.Scatter(x=np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], seasonal_dynamic_regression_predicts),
                            name="Seasonal dynamic regression"))
    fig.add_trace(go.Scatter(x = np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], ets_predicts),
                             name="ETS"))
    fig.add_trace(go.Scatter(x=np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], arima_predicts),
                             name="ARIMA"))
    fig.add_trace(go.Scatter(x = np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], prophet_predicts),
                             name="Prophet"))
    fig.add_trace(go.Scatter(x = np.arange(len(train) - 1, len(train) + len(test)),
                             y=np.append([train[len(train) - 1]], naive_predicts),
                             name="Naive"))
    fig.show()
