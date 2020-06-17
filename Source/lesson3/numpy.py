import numpy as np

# CGenerate rendom array using numpy random
rarray = np.random.random(20)
print(rarray)


# Reshape the array to be 4 rows and 5 columns if not already 4*5
rarray.reshape(4,5)

#find row 1 max value
r1max = np.max(rarray[1:])
print(r1max)
#replace row 1 max value with 0
rarray[rarray == r1max] = 0

#find row 2 max value
r2max = np.max(rarray[2:])
print(r2max)
#replace row 2 max value with 0
rarray[rarray == r2max] = 0

#find row 3 max value
r3max = np.max(rarray[3:])
print(r3max)
#replace row 3 max value with 0
rarray[rarray == r3max] = 0

#find row 4 max value
r4max = np.max(rarray[4:])
print(r4max)
#replace row 4 max value with 0
rarray[rarray == r4max] = 0

print(rarray)
