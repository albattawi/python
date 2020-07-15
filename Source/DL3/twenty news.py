from sklearn.datasets import fetch_20newsgroups
from sklearn import model_selection, preprocessing
from keras import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.embeddings import Embedding
import warnings
warnings.filterwarnings("ignore")

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

sentences = twenty_train.data
y = twenty_train.target

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# # print(input_dim)
model = Sequential()
model.add(layers.Dense(300,input_dim=2000, activation='relu'))
model.add(layers.Dense(300,activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)


max_review_len= 100
vocab_size= len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len,trainable=False))
#model.add(embedding_layer)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

print(model.summary())
