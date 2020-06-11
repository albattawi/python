# let user input the string
str1 = input('Please, Enter a String: ')

#verify User didn't enter empty string (should only enter Valid string)
if not str1.isspace() and str1:
    #convert word to list of characters
    str1 = list(str1)
    print(str1)
    # find the middle length of list and get it as int
    n = int(len(str1)/2)
    print (n)
    #remove item from string according to index by using pop function
    str1.pop(n)

    # Check if index +1 out of range
    if (n+1) < len(str1):
        str1.pop(n+1)
    print(str1)
    # Reverse list and converted to string
    str1= ''.join(reversed(str1))

    #print final result
    print(str1)
else:
    print('Please, Enter a valid string!!!')