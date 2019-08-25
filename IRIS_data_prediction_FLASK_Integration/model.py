# Himanshu Tripathi

# import necessary libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# read data file
iris_data = pd.read_csv('iris.data',sep=',',names=['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm','target'])

# change catogerical values
data = iris_data
le = LabelEncoder()
data['target'] = le.fit_transform(data['target'])
# data.head()

# model creation
X = data.iloc[:,[0,1,2,3]].values
y = data.iloc[:,-1].values

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)

# logistic Regression
model = LogisticRegression()
model.fit(X_test,y_test)

# save model
pickle.dump(model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

# predict 
#y_pred = model.predict(X_test)

#cm = confusion_matrix(y_test,y_pred)
#accuracy = accuracy_score(y_test,y_pred)
#