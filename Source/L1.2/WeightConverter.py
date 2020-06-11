
l1 = input("please enter all students weight in lb separated with ',': ").strip()
# convert the string to list
l1 = l1.split(',')
print(l1)

l2 = []
#define variable to save the convert value from lb to kg
lb = float(0.453592)

for x in range(len(l1)):
    #find the weight in kg
    kg = float(l1[x]) * lb
    #enter the kg value to the list l2
    l2.append(kg)
#loop for print
for x in range(len(l1)):
    print('Student#: ' + str(x+1) + ': Weight in lb ' + str(l1[x]) + ' = ' + str(l2[x]) + ' kg')

#print('Students Weight with Pounds lb: ')
#print(l1)
#print('*******************************\n')
#print('Students Weight with kg :')
#print(l2)
