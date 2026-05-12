ass 1

import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)

print(df.info())
print(df.dtypes)
print(df.describe())
print(df.columns)
print(df.isnull())

print(df.isnull().sum())
print(df.notnull())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Name'] = df['Name'].fillna("Unknown")
df['Embarked'] = df['Embarked'].fillna("S")

df['Age'] = df['Age'].fillna(df['Age'].mode())

df = df.ffill()
print(df.head(6))

df = df.bfill()
print(df.head(6))

df['Age'] = df['Age'].ffill()
df.dropna(inplace=True)

print(df.duplicated())
df.drop_duplicates(inplace=True)

df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(float)

df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
print(df[['Age']].head())

print(df.columns)

df_original = pd.read_csv("titanic.csv")

df['Sex'] = df_original['Sex']

df['Sex'] = df['Sex'].astype(str).str.lower().str.strip()

print(df['Sex'].unique())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print(df[['Sex']].head())
print(df['Sex'].isnull().sum())
