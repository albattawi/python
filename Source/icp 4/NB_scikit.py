import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

glass = datasets.load_iris('train_glass.csv')
glass_reshape = np.reshape('a',(-1,1))

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split('Na','Mg', test_size=0.4, random_state=0)

print(X_test)
# #Create a Gaussian Classifier
gnb = GaussianNB()
# #Train the model using the training sets
gnb.fit(X_train, y_train)
#
# Predict the response for test dataset
y_pred = gnb.predict(X_test)


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))