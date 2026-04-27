import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
print(df.head())

print(df.isnull().sum())

for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include=['object','string']):
    df[col] = df[col].fillna(df[col].mode()[0])

sns.boxplot(x='sex', y='age', data=df, hue='survived')
plt.title("Age Distribution by Gender and Survival")
plt.show()

sns.countplot(x='survived', data=df)
plt.show()

sns.histplot(df['age'], bins=10)
plt.show()


sns.scatterplot(x='age', y='fare', data=df)
plt.show()

sns.boxplot(x='sex', y='age', data=df, hue='survived')

print(df.head())