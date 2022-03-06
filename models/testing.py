import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from dynamic_regression import DynamicRegression
from models import SktimeETS, SktimeARIMA, SktimeLinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

dataset = pd.read_csv("/Users/beast-sl/source/repos/coursework-3/Dataset/Train/Daily-train.csv")
testing_series = dataset[dataset["V1"] == "D1"].squeeze().dropna().drop(index='V1')
testing_series.index = range(len(testing_series))
testing_series = testing_series.astype(np.float)
train, test = temporal_train_test_split(testing_series, test_size=5)

print("Testing Dynamic Regression")
model = DynamicRegression()
model.fit(train)
dynamic_regression_predicts = model.predict(train)
print(f"MSE: {mean_squared_error(dynamic_regression_predicts, test)}")

print("Testing ETS")
model = SktimeETS(auto=True, n_jobs=-1)
model.fit(train)
ets_predicts = model.predict(fh=np.arange(1, 6))
print(f"MSE: {mean_squared_error(ets_predicts, test)}")

print("Testing ARIMA")
model = SktimeARIMA(order=(1, 1, 0), suppress_warnings=True)
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
fig.add_trace(go.Scatter(y=dynamic_regression_predicts, name="Dynamic regression"))
fig.add_trace(go.Scatter(y=ets_predicts, name="ETS"))
fig.add_trace(go.Scatter(y=arima_predicts, name="ARIMA"))
fig.add_trace(go.Scatter(y=linear_regression_predicts, name="Linear regression"))
fig.show()
