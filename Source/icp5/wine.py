import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
import matplotlib. pyplot as plt

# Using pandas module to read csv file as wines
wines = pd.read_csv('winequality-red.csv')

# Check the Null values in wines; will return True if Null exists and False if not
isnullf = wines.isnull()
print(isnullf)
# Find the total number of Null values we have in DF and print it
print(str(isnullf.sum()))

#Remove rows with empty values if found any using built in function drop na
if True in wines.isnull():
    wines = wines.dropna()

    #new_wines = modifieddf_wines
else:
    wines

# Convert final results to DataFrame
df_wines = pd.DataFrame(wines)

# Create a target label as quality_label and categories it as;
# Low for less or equal to 5;
# Medium between 6 and 7;#
# High for more then 7
wines['quality_label'] = wines.quality.apply(lambda q: 'low' if q <= 5 else 'medium' if q <= 7 else 'high')

# x as feature and y as target
x = df_wines[['alcohol']]
y = df_wines[['quality']]

# add constant as x
x = sm.add_constant(x)

#Create OLS Model as Ordinary Least square
model = sm.OLS(y,x).fit()
prediction = model.predict(x)

# Because function RMSE are not already built in function in sklearn module,
# then we have to find the MSE first and then use square method from numpy module
MSE = mean_squared_error(prediction,y)
RMSE = np.sqrt(MSE)
print(model.summary())
prediction

print('MRE:' + str(MSE))
print('RMSE: ' + str(RMSE))
