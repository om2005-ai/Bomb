import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("BostonHousing.csv")

print(df.info())
print(df.describe())

df = df.fillna(df.mean())

df_x = df.drop("medv", axis=1)
df_y = df["medv"]

reg.fit(x_train, y_train)

print(reg.coef_)

y_pred = reg.predict(x_test)
print(y_pred)

print("Predicted value (3rd row):", y_pred[2])
print("Actual value:", y_test.iloc[2])

print(np.mean((y_pred - y_test) ** 2))

print(mean_squared_error(y_test, y_pred))