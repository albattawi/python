# create a dictionary to save each word as in file
wordcount = {}

# open a text file and read lines
with open(r'C:\Users\badri\.PyCharmCE2019.3\config\scratches\ReadFile.txt','r') as file:
    for word in file.read().split():
        #verify the word is not already in the dictionary wordcount
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
print(wordcount)
#with open(r'C:\Users\badri\.PyCharmCE2019.3\config\scratches\ReadFile.txt','w') as file:
try:
    file = open(r'C:\Users\badri\.PyCharmCE2019.3\config\scratches\WriteFile.txt','x')
except FileExistsError:
   file = open(r'C:\Users\badri\.PyCharmCE2019.3\config\scratches\WriteFile.txt', 'w')

for item in wordcount.items():
    print("{}: {}".format(*item))
    file.write("{}: {}\n".format(*item))
file.close()