import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

train_glass = pd.read_csv("./glass.csv")
test_glass = pd.read_csv("./glass.csv")
print(train_glass.groupby('Mg')[['Na']].mean())
x = train_glass.drop('Mg', axis=1)
y = train_glass['Na']
x_test = test_glass.drop('Mg',axis=1).copy()
print(train_glass[train_glass.isnull().any(axis=1)])
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=0)


##SVM
svc = SVC()
svc.fit(x,y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x, y_train) * 100, 2)
print('svm accuracy is:', acc_svc)





