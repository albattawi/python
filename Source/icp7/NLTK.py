# importing the libraries
import urllib.request
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
import nltk
# Reading input.text file and save it in my_text to work wit it
with open('input.txt','r', encoding="utf-8") as f:
    my_text = f.read()
    f.close()


# Apply Tokenization on my_text as stokens
stokens = nltk.sent_tokenize(my_text)
file = open('Stokens.txt','w',encoding="utf-8")
for s in stokens:
    file.write(s)
    file.write('\n')

print('Tokenization as STokens saved in file:' , file.name)
file.close()


# Apply Tokenization on my_text as wtokens
wtokens = nltk.word_tokenize(my_text)
# file = open('Wtokens.txt','w',encoding="utf-8")
# for w in wtokens:
#     file.write(w)
#     file.write('\n')

print('Tokenization as WTokens saved in file:')
print(wtokens)
# file.close()


# Applying POS - Part of Speech tagging
print('\nPOS - Part of Speech tagging\n-------------------------------------')
print(nltk.pos_tag(wtokens))

#
#Applying Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer


pStemmer = PorterStemmer()
print('\npStemmer\n-------------------------------------')
for w in wtokens:
    print(pStemmer.stem(w))


lStemmer = LancasterStemmer()
print('\nlStemmer\n-------------------------------------')
for w in wtokens:
    print(lStemmer.stem(w))


sStemmer = SnowballStemmer('english')
print('\nEnglish - sStemmer\n-------------------------------------')
for w in wtokens:
    print(sStemmer.stem(w))


# Applying Lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print('\nlemmatizer\n-------------------------------------')
print(lemmatizer.lemmatize('google'))
print(lemmatizer.lemmatize('googles'))
print(lemmatizer.lemmatize('googling'))#Applying Trigram
count_vect= CountVectorizer()


from collections import Counter
# #create a Trigram Function
def get_ngrams_count(words, n):
    # turn the list into a dictionary with the counts of all n-grams
    return Counter(zip(*[words[i:] for i in range(n)]))
#Call trigrams function
trigrams = get_ngrams_count(wtokens,3)

print('\nTrigram\n-------------------------')
for t in trigrams:
    print(t)
file = open('Trigrams.txt','w',encoding="utf-8")
file.write(str(trigrams))
print('Trigram N=3 saved in file: ',file.name)

# Applying NER - Named Entity Recognition
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

print('\nNamed Entity Recognition\n-------------------------')
NER = str(ne_chunk(pos_tag(wordpunct_tokenize(my_text))))
file = open('NER.txt','w',encoding="utf-8")
file.write(NER)
print('Named Entity Recognition saved in file: ',file.name)

