from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#importing our cancer dataset
dataset = pd.read_csv(r'C:\Users\badri\PycharmProjects\untitled11\Breas Cancer.csv')
del dataset['Unnamed: 32']

X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values
print(dataset.head())
print("Cancer data set dimensions : {}".format(dataset.shape))


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# dataset = pd.read_csv("Breas Cancer.csv", header=1).values
# X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:], dataset[:20],
#                                                     test_size=0.25, random_state=87)

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))
