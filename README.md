# Linear Regration: 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df

df.info()

df.describe()

df.columns

sns.distplot(df['Price'])

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X_train
y_train
X_test
y_test

## Creating and Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.coef_
X.columns

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

lm.score(X_test,y_test)

predictions = lm.predict(X_test)
predictions

y_test

plt.scatter(y_test,predictions)



## -------Decision tree-------

import pandas as pd
import numpy as np
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df=pd.read_csv('/content/drive/MyDrive/Data mining & ML/heart_failure.csv')
df

df.isnull().values.any()
df.describe()

x = df.drop(['DEATH_EVENT'], axis = 1)
x

y = df['DEATH_EVENT']
y

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=.3,random_state=1)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(xtrain,ytrain)

predictions = dtc.predict(xtest)

predictions

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(ytest,predictions))

print(confusion_matrix(ytest,predictions))

dtc.score(xtest,ytest)

## Tree Visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[:-1])
features

print(dtc.predict([[50, 1, 181, 0, 20, 0, 210000, 1.2, 153, 0, 0, 5]]))
print ("'0' Dead with heart fail")
print ("'1' Not dead")


## random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xtrain,ytrain)

rfc_pred = rfc.predict(xtest)

print(confusion_matrix(ytest,rfc_pred))

rfc.score(xtest,ytest)

print(rfc.predict([[50, 1, 181, 0, 20, 0, 210000, 1.2, 153, 0, 0, 5]]))
print ("'0' Dead with heart fail")
print ("'1' Not dead")


