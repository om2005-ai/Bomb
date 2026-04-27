import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

print(df.head())

print(df.isnull().sum())

X = df.iloc[:, 1:-1].values   
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

f_measure = 2 * (precision * recall) / (precision + recall)
print("F-measure:", f_measure)