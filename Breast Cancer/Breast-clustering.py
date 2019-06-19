#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = 'Colon.csv'
namesUrl = "names-colon.txt"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
names = pd.read_csv(namesUrl, nrows=1).columns.tolist()
dataframe = pd.read_csv(url, names=names)
# replace missing values
dataframe= dataframe.replace('Tumor',1)
dataframe= dataframe.replace('Normal',0)


 
#The features X are everything except for the class.
X = np.array(dataframe.drop(['class'], 1))  
X = preprocessing.scale(X)

# Y is just the class or the diagnosis column 
y = np.array(dataframe['class']) 

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(y)
print(correct)
print(len(X))