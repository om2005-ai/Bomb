
import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
df.head()

print(df.columns)

df = df.drop(columns=['Id'])

df.columns = ['sepal-length','sepal-width','petal-length','petal-width','class']

print(df.head())

df.tail()

df.info()

df.describe()

print(df.shape)
print(df.isnull().sum())

grouped = df.groupby('class')
grouped.describe()

print(grouped.mean())
print(grouped.median())
print(grouped.std())
print(grouped.min())
print(grouped.max())

grouped.quantile([0.25, 0.5, 0.75])

