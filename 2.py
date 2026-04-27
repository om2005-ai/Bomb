import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("student.csv")
print(df.head())
print(df.tail())

print(df.info())
print(df.describe())

print(df.isnull().sum())

df.fillna({col: df[col].mean()}, inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, len(outliers))

    for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df[col] = np.where(df[col] > upper, upper, df[col])
    df[col] = np.where(df[col] < lower, lower, df[col])

    print(df.columns)

    plt.hist(df["studytime"])
    plt.title("Before Transformation")
    plt.show()

    df["Log_studytime"] = np.log1p(df["studytime"])

    plt.hist(df["Log_studytime"])
    plt.title("After Transformation")
    plt.show()