from keras import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.embeddings import Embedding
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print('Null Values: ',df.isnull().values.any())
print(df.shape)
print(df.head())

def textclean(Textdata):
    documents = []

    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stemmer = WordNetLemmatizer()
    stop_word = set(stopwords.words("english"))
    for sen in range(0, len(Textdata)):
    # Remove all the special characters
    #   document = re.sub(r'\W', ' ', Textdata.re.UNICODE)

    # remove all single characters
      document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
      document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
      document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
      document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
      document = document.lower()

    # Lemmatization
      document = document.split()
      document = [stemmer.lemmatize(word) for word in document]
      document = ' '.join(document)
    documents.append(document)

    return documents


sentences = df['review'].values
y = df['label'].values

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.35, random_state=2000)

# Number of features
# # print(input_dim)
model = Sequential()
model.add(layers.Dense(300,input_dim=2000, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)


max_review_len= 2000
vocab_size= len(tokenizer.word_index)+1

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len,trainable=False))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()



print('Predect: ',model.predict(X_test))