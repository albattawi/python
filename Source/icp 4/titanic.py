import numpy as np
import pandas as pd

train_preprocessed = pd.read_csv("./train_preprocessed.csv")
test_preprocessed = pd.read_csv("./test_preprocessed.csv")
print(train_preprocessed.groupby('Sex')[['Survived']].mean())
x_train = train_preprocessed.drop('Survived', axis=1)
y_train = train_preprocessed['Survived']
x_test = test_preprocessed.drop('PassengerId',axis=1).copy()
print(train_preprocessed[train_preprocessed.isnull().any(axis=1)])
