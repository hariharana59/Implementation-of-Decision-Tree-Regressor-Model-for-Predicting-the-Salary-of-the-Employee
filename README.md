# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 . Start

STEP 2 . attach the given data file

STEP 3 . now find the satisfaction level of employee data

STEP 4 .find the accuracy and new predict value

STEP 5 . end.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Hariharan A
RegisterNumber: 212223110013
*/

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform (data["Position"]) 
data.head()

x=data[["Position", "Level"]] 
y=data["Salary"]

from sklearn.model_selection import train_test_split 
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score (y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
![Screenshot 2024-10-07 111249](https://github.com/user-attachments/assets/e8dcb5bc-abe2-40fe-9317-68172413d92c)
![Screenshot 2024-10-07 111300](https://github.com/user-attachments/assets/0865542e-28f5-4ced-8cab-4fc0b257ac1a)
![Screenshot 2024-10-07 111310](https://github.com/user-attachments/assets/6fac809b-1f0e-4713-969d-5b3610a085ce)
![Screenshot 2024-10-07 111330](https://github.com/user-attachments/assets/3d349036-75ba-411c-9196-c10782baafb6)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
