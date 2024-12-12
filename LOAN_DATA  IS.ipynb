# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:18:52 2024

@author: sriha
"""

import pandas as pd
df=pd.read_csv("C:\\Users\\sriha\\OneDrive\\Desktop\\pb excel\\loan_data_iNTERNSHIP STUDIO.csv")

df.isnull().sum()

df.fillna(method="ffill",inplace=True)
df.fillna(method="bfill",inplace=True)

df1=pd.get_dummies(df,columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area","Loan_Status"])

df1.dtypes
df1.drop("Gender_Female",axis=1,inplace=True)
df1.drop("Education_Graduate",axis=1,inplace=True)
df1.drop("Self_Employed_No",axis=1,inplace=True)
df1.drop("Property_Area_Rural",axis=1,inplace=True)
df1.drop("Loan_Status_N",axis=1,inplace=True)
df1.drop("Dependents_3+",axis=1,inplace=True)


x=df1.iloc[:,2:15]
y=df1.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=(0))

from sklearn.linear_model import LogisticRegression
m1=LogisticRegression()
m1.fit(x_train, y_train)
y_pred=m1.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
acc=accuracy_score(y_test, y_pred)
c=confusion_matrix(y_test,y_pred)
print(acc)
print(c)
