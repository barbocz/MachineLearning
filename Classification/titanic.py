import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
titanic_data=pd.read_csv('titanic_data.csv')
# print(titanic_data.head(10))
# # pl=sns.countplot(x='Survived',hue='Sex',data=titanic_data)
# titanic_data['Age'].plot.hist()

# titanic_data.info()
# sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')

titanic_data.drop('Cabin',axis=1,inplace=True)
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')

# sns.boxplot(x='Pclass',y='Age',data=titanic_data)
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
Pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data.drop(['PassengerId','Pclass','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# print(titanic_data.head(5))
# plt.show()
X=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)
logmodel=LogisticRegression(max_iter=3000)
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))