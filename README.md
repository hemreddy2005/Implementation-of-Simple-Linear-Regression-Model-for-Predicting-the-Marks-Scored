# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.

2.Import the standard Libraries.

3.Set variables for assigning dataset values.

4.Import linear regression from sklearn.

5.Assign the points for representing in the graph.

6.Predict the regression for marks by using the representation of the graph.

7.End the program. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hemanth Kumar R
RegisterNumber:  212223040065
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
```
```
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  
```
## Output:

df.head()

![Img 1](https://github.com/user-attachments/assets/4d2b8c72-002a-4022-8714-a7f3cc2a1d84)

df.tail()

![Img 2](https://github.com/user-attachments/assets/91ab9f11-f306-401f-a296-873a68eaaadc)

Array Value of x

![Img 3](https://github.com/user-attachments/assets/a98b06e4-a0cd-4bdd-ad02-0b847771f995)

Array value of y

![Img 4](https://github.com/user-attachments/assets/3053a226-59f6-41a8-b390-eb5b57ecfb90)

Values of y prediction

![Img 5](https://github.com/user-attachments/assets/058f8571-0703-437c-bf9e-3783772fd4a1)

Array values of Y test

![Img 6](https://github.com/user-attachments/assets/93840ff2-1ff3-4e6b-878f-08c5ae2a9ae4)

Training set graph

![Img 7](https://github.com/user-attachments/assets/5c13c02e-e4d2-45dd-b709-039e52e8edd5)

Test set graph

![Img 8](https://github.com/user-attachments/assets/6fc08e7f-e1b8-4eaf-a9dc-c28477d503d3)

Values of MSE,MAE and RMSE

![Img 9](https://github.com/user-attachments/assets/6c3b265d-59ec-434f-9b9f-4c53d1348981)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
