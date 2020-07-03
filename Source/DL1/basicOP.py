from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



dataset = pd.read_csv("diabetes.csv", header=None).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))

#dataset1 = pd.read_csv("diabetes.csv", header=None).values

# X_train, X_test, Y_train, Y_test = train_test_split(dataset1[:,0:8], dataset1[:,8],
#                                                     test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn1 = Sequential() # create model
my_first_nn1.add(Dense(25, input_dim=8, activation='relu')) # hidden layer
my_first_nn1.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn1_fitted = my_first_nn1.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print('\n**************************************************\nNew Results after change Dense from 20 to 25**************************************************\n')
print(my_first_nn1.summary())
print(my_first_nn1.evaluate(X_test, Y_test))
