from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


dataset = pd.read_csv("Breas Cancer.csv", header=1).values

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:13], dataset[:,10],
                                                    test_size=0.25, random_state=87)
print(dataset.shape)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=11, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
