
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:29:23 2018

@author: IKRAM
"""

# Bagged SVM for Classification
import pandas 
from sklearn import model_selection,metrics,svm
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import NuSVC


#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
url = 'Colon.csv'
namesUrl = "names-colon.txt"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
names = pandas.read_csv(namesUrl, nrows=1).columns.tolist()
dataframe = pandas.read_csv(url, names=names)
# replace missing values
dataframe= dataframe.replace('Tumor',1)
dataframe= dataframe.replace('Normal',0)


 
#The features X are everything except for the class.
X = np.array(dataframe.drop(['val'], 1))  

# Y is just the class or the diagnosis column 
y = np.array(dataframe['val']) 


X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2) 

clf_lsvc = LinearSVC()
clf = svm.SVC() 
clf_nu = NuSVC()

clf.fit(X_train, y_train) 
clf_nu.fit(X_train, y_train)
clf_lsvc.fit(X_train, y_train)

accuracy_svm = clf.score(X_test, y_test) 
accuracy_nu = clf_nu.score(X_test, y_test) 
accuracy_lsvc = clf_lsvc.score(X_test, y_test) 

print("SVC", accuracy_svm)
print("NuSVC", accuracy_nu)
print("Linear SVC", accuracy_lsvc)

# Cross Validation
predicted_svm = cross_val_predict(clf, X, y, cv=10)
predicted_nu = cross_val_predict(clf_nu, X, y, cv=10)
predicted_lsvc = cross_val_predict(clf_lsvc, X, y, cv=10)

print("SVC Cross-validation",metrics.accuracy_score(y, predicted_svm))
print("NuSVC Cross-validation",metrics.accuracy_score(y, predicted_nu))
print("Linear Cross-validation",metrics.accuracy_score(y, predicted_lsvc))

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
clf_lsvc = LinearSVC(random_state=0)
num_trees = 100
model = BaggingClassifier(base_estimator=clf_lsvc, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print(results.mean())
"""
SVC 0.72
NuSVC 0.72
Linear SVC 0.8
SVC Cross-validation 0.645161290323
NuSVC Cross-validation 0.645161290323
Linear Cross-validation 0.822580645161
SVM + Bagging 0.854761904762
"""