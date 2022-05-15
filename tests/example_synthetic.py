import numpy as np
from pprint import pprint
from sktime.forecasting.model_selection import temporal_train_test_split
from models.dynamic_regression import DynamicRegression
from models.models import SktimeETS, SktimeARIMA, SktimeLinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def plot_predictions(train, test):
    print("Testing Dynamic Regression")
    model = DynamicRegression(fh=np.arange(1, 6))
    status, history = model.fit(train)
    dynamic_regression_predicts = model.predict(train)
    print(f"Final params: {model.params}")
    print(f"MSE: {mean_squared_error(dynamic_regression_predicts, test)}")

    print("Testing ETS")
    model = SktimeETS(auto=True, n_jobs=-1)
    model.fit(train)
    ets_predicts = model.predict(fh=np.arange(1, 6)).squeeze()
    print(f"MSE: {mean_squared_error(ets_predicts, test)}")

    print("Testing ARIMA")
    model = SktimeARIMA(suppress_warnings=True)
    model.fit(train)
    arima_predicts = model.predict(fh=np.arange(1, 6)).squeeze()
    print(f"MSE: {mean_squared_error(arima_predicts, test)}")

    print("Testing Linear Regression")
    model = SktimeLinearRegression()
    model.fit(train)
    linear_regression_predicts = model.predict(fh=np.arange(1, 6)).squeeze()
    print(f"MSE: {mean_squared_error(linear_regression_predicts, test)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(len(train) + len(test)), y=np.append(train, test), name="Test"))
    fig.add_trace(go.Scatter(x = np.arange(len(train), len(train) + len(test)),
                             y=dynamic_regression_predicts,
                             name="Dynamic regression"))
    fig.add_trace(go.Scatter(x = np.arange(len(train), len(train) + len(test)), y=ets_predicts, name="ETS"))
    fig.add_trace(go.Scatter(x = np.arange(len(train), len(train) + len(test)), y=arima_predicts, name="ARIMA"))
    fig.add_trace(go.Scatter(x = np.arange(len(train), len(train) + len(test)),
                             y=linear_regression_predicts,
                             name="Linear regression"))
    fig.show()


if __name__ == '__main__':
    # train = np.array([1] * 100)
    # test = np.array([1] * 5)
    # plot_predictions(train, test)

    print("UPWARD TREND")
    train = np.arange(100)
    test = np.arange(100, 105)
    plot_predictions(train, test)
    print()

    print("DOWNWARD TREND")
    train = np.arange(100, -1, -1)
    test = np.arange(-1, -6, -1)
    plot_predictions(train, test)

    print("TREND CHANGE")
    train = np.append(np.arange(50), np.arange(48, -1, -1))
    test = np.arange(-1, -6, -1)
    plot_predictions(train, test)

    print("LEAP")
    train = np.append([0] * 50, [50] * 50)
    test = np.array([50] * 5)
    plot_predictions(train, test)
    