from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
count_vect= CountVectorizer()
X_train_counts= count_vect.fit_transform(twenty_train.data)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)

from sklearn.svm import LinearSVC

# Instantiate the estimator
clf_svc = LinearSVC()

# Fit the model with data (aka "model training")
clf_svc.fit(X_train_tfidf, twenty_train.target)

# Predict the response for a new observation
y_pred = clf_svc.predict(X_test_tfidf)
print("Predicted Class Labels:",y_pred)


# Predict the response score for a new observation

acc_svc = clf_svc.score(X_train_tfidf, twenty_train.target)

print(acc_svc)


count_vect2= CountVectorizer(analyzer='word', ngram_range=(1, 2),max_df=1.0, min_df=1, max_features=None)
count_vect2 = count_vect2.fit(twenty_train.data)
X_train_counts2= count_vect2.fit_transform(twenty_train.data)
print(count_vect2.get_feature_names())
print('\n')


count_vect= CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2),max_df=1.0, min_df=1, max_features=None)
count_vect = count_vect2.fit(twenty_train.data)
X_train_counts= count_vect.fit_transform(twenty_train.data)

print(count_vect.get_feature_names())

