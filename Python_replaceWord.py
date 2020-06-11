#Let user input any sentence
pystr = input('Please Enter your python sentence: ')

#verify the sentence have the word python at leat once

if 'python' in pystr:
    pystrlist = pystr.split(' ')
    print (pystrlist)
    for x in range(len(pystrlist)):
        if 'python' in pystrlist[x]:
            pystrlist[x] = 'pythons'
    print(pystrlist)
    print (' '.join(pystrlist))
else:
    print('Please make sure you have word (python) at least once in your sentence!!')