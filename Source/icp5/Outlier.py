import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib. pyplot as plt

train = pd.read_csv('train.csv')
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)


df_houses = pd.DataFrame(train)

x = df_houses['GarageArea']
y = df_houses['SalePrice']

sns.scatterplot(x,y)
plt.show()

#Identifying Outliers with Skewness
print('GarageArea Skewness')
print(df_houses['GarageArea'].skew())
df_houses['GarageArea'].describe()
print()
print('SalePrice Skewness')
print(df_houses['SalePrice'].skew())
df_houses['SalePrice'].describe()

print('GarageArea Quantile')
print(df_houses['GarageArea'].quantile(0.17))
print(df_houses['GarageArea'].quantile(0.0187))

print('GarageArea SalePrice')
print(df_houses['SalePrice'].quantile(0.17))
print(df_houses['SalePrice'].quantile(0.187))
print()
#Now we will remove the outliers, as shown in the lines of code below. Finally, we calculate the skewness value again, which comes out much better now.
print('Remove Outliers for GarageArea')
df_houses["GarageArea"] = np.where(df_houses["GarageArea"] <0.179, 0.179,df_houses['GarageArea'])
df_houses["GarageArea"] = np.where(df_houses["GarageArea"] >0.890, 0.890,df_houses['GarageArea'])
print(df_houses['GarageArea'].skew())
print()
print('Remove Outliers for SalePrice')
df_houses["SalePrice"] = np.where(df_houses["SalePrice"] <119000.0, 119000.0,df_houses['SalePrice'])
df_houses["SalePrice"] = np.where(df_houses["SalePrice"] >120916.0, 120916.0,df_houses['SalePrice'])
print(df_houses['SalePrice'].skew())


#Removing the GarageArea Outliers
print('Removing the GarageArea Outliers')
index = df_houses[(df_houses['GarageArea'] >= 0.17)|(df_houses['GarageArea'] <= 1.88)].index
df_houses.drop(index, inplace=True)
df_houses['GarageArea'].describe()
print()
#Removing the SalePrice Outliers
print('Removing the SalePrice Outliers')
index = df_houses[(df_houses['SalePrice'] >= 119000.0)|(df_houses['SalePrice'] <= 120916.50)].index
df_houses.drop(index, inplace=True)
df_houses['SalePrice'].describe()


