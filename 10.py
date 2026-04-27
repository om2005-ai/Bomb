import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")   
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
sns.boxplot(y=df['SepalLengthCm'])
plt.subplot(2,2,2)
sns.boxplot(y=df['SepalWidthCm'])
plt.subplot(2,2,3)
sns.boxplot(y=df['PetalLengthCm'])
plt.subplot(2,2,4)
sns.boxplot(y=df['PetalWidthCm'])
plt.suptitle("Boxplots of Features")
plt.show()

sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.title("Sepal Length Distribution by Species")
plt.show()

print(df.head())
