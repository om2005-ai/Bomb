import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
print(df.head())

print(df.info())
print(df.describe())
print(df.columns)


print(df.isnull().sum())

for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include=['object','string']):
    df[col] = df[col].fillna(df[col].mode()[0])

sns.histplot(df['fare'], bins=10, kde=False)
plt.title("Fare Distribution")
plt.show()

sns.histplot(df['age'], bins=10, kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x='sex', data=df)
plt.title("Gender Count")
plt.show()

sns.boxplot(x='sex', y='age', data=df)
plt.title("Box Plot")
plt.show()

sns.violinplot(x='sex', y='age', data=df)
plt.title("Violin Plot")
plt.show()

sns.stripplot(x='sex', y='age', data=df, jitter=True)
plt.title("Strip Plot")
plt.show()

sns.stripplot(x='sex', y='age', data=df, jitter=True)
plt.title("Swarm Plot")
plt.show()

sns.jointplot(x='age', y='fare', data=df, kind='scatter')

sns.rugplot(df['fare'])
plt.title("Rug Plot")
plt.show()

sns.barplot(x='sex', y='age', data=df)
plt.title("Average Age by Gender")
plt.show()

sns.countplot(x='sex', data=df)
plt.title("Gender Count")
plt.show()

sns.stripplot(x='sex', y='age', data=df, jitter=True)
plt.title("Strip Plot")
plt.show()

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

