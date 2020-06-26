import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Read our data from csv file to dataset
dataset = pd.read_csv('CC.csv')
print('Dataset before Check Null: ', str(dataset.shape))

# Check the Null values in datset; will return True if Null exists and False if not
# Find the total number of Null values we have in dataset and print it
print('Number of Null Data in Dataset: ',dataset.isnull().sum().sum())

# Remove rows with empty values if found any using built in function drop na
dataset = dataset.dropna(how ='any')

# Verify dataset dropped all na rows and print the new shape of it
print('\nNumber of Null Data in Dataset after drop: ', dataset.isnull().sum().sum())
print('Dataset after Check Null: ', str(dataset.shape))
x = dataset.iloc[:,[1,3,6,13,14,15]]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)

# Visualize original dataset
sns.FacetGrid(dataset, hue="TENURE", size=6).map(plt.scatter, "BALANCE","CASH_ADVANCE").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=6).map(plt.scatter, "CREDIT_LIMIT","PURCHASES").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=6).map(plt.scatter, "PAYMENTS","MINIMUM_PAYMENTS").add_legend()
plt.show()

# standarized
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)


# Using Kmeans will calculate the SSE (sum square error) for the range of k as from 1 to 10
k_range = range(1,40)
sse = []
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(dataset[['BALANCE','CASH_ADVANCE','CREDIT_LIMIT','PURCHASES','PAYMENTS','MINIMUM_PAYMENTS']])
    sse.append(km.inertia_)
print('SSE LIST')
print(sse)
# Print centeroid
#print(km.cluster_centers_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_range,sse)
plt.show()

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Kmeans Score:', score)

# Use PCA function to reduce data dimensions and use it on scaled dataset in this case is (x_scaled)
pca = PCA(2)
pca.fit(X_scaled)
pca_data = pca.transform(X_scaled)
print(pca_data)

#Visiulize the PCA on dataset
plt.plot(pca_data)
plt.show()

k_range = range(1,40)
sse = []
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(pca_data)
    sse.append(km.inertia_)
print('SSE LIST')
print(sse)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
plt.plot(k_range,sse)
plt.show()

y_cluster_kmeans = km.predict(pca_data)
score = metrics.silhouette_score(pca_data, y_cluster_kmeans)
print('PCA Score:', score)