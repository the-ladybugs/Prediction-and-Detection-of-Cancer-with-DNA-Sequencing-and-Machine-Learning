#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cross_validation import train_test_split
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = 'Colon.csv'
namesUrl = "names-colon.txt"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
names = pd.read_csv(namesUrl, nrows=1).columns.tolist()
dataframe = pd.read_csv(url, names=names)
# replace missing values
dataframe= dataframe.replace('Tumor',1)
dataframe= dataframe.replace('Normal',0)
dataframe = dataframe[dataframe.val != 0]


#The features X are everything except for the class.
X = np.array(dataframe.drop(['val'], 1))  
X = preprocessing.scale(X)

# Y is just the class or the diagnosis column 
Y = np.array(dataframe['val']) 

training_data, test_data, training_target, test_target  = train_test_split(X, Y, test_size=.3)
# Identify number of clusters using the elbow method
K=range(1,5)
distortions=[]

for k in K:
    kmeanModel = KMeans(n_clusters=k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 800, n_init = 30, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 100], s = 100, c = 'green', label = 'Küme 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 100], s = 100, c = 'blue', label = 'Küme 2')
plt.xlabel('Colon Cancer - Number of Clusters')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'yellow', label = 'Küme Merkezleri')
plt.legend()
plt.show()
# Visualize the elbow

k = 2

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(K, distortions)
ax.plot(K[(k-1)], distortions[(k-1)], marker='x', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Colon Cancer - Number of Clusters')
plt.ylabel('Average Distance')
plt.show()

kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
labels = kmeanModel.labels_



