import pandas as pd
import numpy as np

df = pd.read_csv("loan.csv")
df.head()

df.columns

df = df.dropna(subset=['age', 'Principal'])

grouped = df.groupby('age')

print(grouped['Principal'].mean())

print(grouped['Principal'].agg(['mean','median','min','max','std']))

print(grouped['Principal'].quantile([0.25, 0.5, 0.75]))

print(grouped.describe())

print(df.isnull().sum())