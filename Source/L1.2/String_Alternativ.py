# define a New fuction name string_alternative
def string_alternative(str1):
    # using slice notation to print every other charater as slice notation work as [start:stop:step]
    print(str1[::2])
    return ;
#Define Main program
def main():
    # let user enter any string
    str1 = input('Please, Enter String: ')
    string_alternative(str1)
# Call Main Function
main()


