import pandas as pd
import numpy as np
from pprint import pprint
from sktime.forecasting.model_selection import temporal_train_test_split
from models.dynamic_regression import DynamicRegression
from models.models import SktimeETS, SktimeARIMA, SktimeLinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

if __name__ == '__main__':
    dataset = pd.read_csv("/Users/beast-sl/source/repos/coursework-3/Dataset/Train/Monthly-train.csv")
    testing_series = dataset[dataset["V1"] == "M100"].squeeze().dropna().drop(index='V1')
    testing_series.index = range(len(testing_series))
    testing_series = testing_series.astype(np.float)
    train, test = temporal_train_test_split(testing_series, test_size=5)

    # print("Testing Dynamic Regression")
    # model = DynamicRegression(fh=np.arange(1, 6))
    # status, history = model.fit(train, trace=True)
    # print(f"Status {status}, history:")
    # pprint(history)
    # dynamic_regression_predicts = model.predict(train)
    # print(f"MSE: {mean_squared_error(dynamic_regression_predicts, test)}")

    print("Testing seasonal Dynamic Regression")
    model = DynamicRegression(fh=np.arange(1, 6), sp=12)
    status, history = model.fit(train, trace=True)
    print(f"Status {status}, history:")
    pprint(history)
    seasonal_dynamic_regression_predicts = model.predict(train)
    print(f"MSE: {mean_squared_error(seasonal_dynamic_regression_predicts, test)}")

    print("Testing ETS")
    model = SktimeETS(auto=True, n_jobs=-1)
    model.fit(train)
    ets_predicts = model.predict(fh=np.arange(1, 6))
    print(f"MSE: {mean_squared_error(ets_predicts, test)}")

    print("Testing ARIMA")
    model = SktimeARIMA(suppress_warnings=True)
    model.fit(train)
    arima_predicts = model.predict(fh=np.arange(1, 6))
    print(f"MSE: {mean_squared_error(arima_predicts, test)}")

    print("Testing Linear Regression")
    model = SktimeLinearRegression()
    model.fit(train)
    linear_regression_predicts = model.predict(fh=np.arange(1, 6))
    print(f"MSE: {mean_squared_error(linear_regression_predicts, test)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test, name="Test"))
    # fig.add_trace(go.Scatter(y=dynamic_regression_predicts, name="Dynamic regression"))
    fig.add_trace(go.Scatter(y=seasonal_dynamic_regression_predicts, name="Dynamic regression"))
    fig.add_trace(go.Scatter(y=ets_predicts, name="ETS"))
    fig.add_trace(go.Scatter(y=arima_predicts, name="ARIMA"))
    fig.add_trace(go.Scatter(y=linear_regression_predicts, name="Linear regression"))
    fig.show()
