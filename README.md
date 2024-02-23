# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Samyuktha S
RegisterNumber: 212222240089
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
```
```
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:

## Dataset

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/a35c9b4a-9cff-4e2b-82e7-998e9d0ad206)

## Head Values

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/1a1b809d-8615-4b4f-bcef-f08dbe353afb)

## Tail Values

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/b4ae5bc2-09a4-4f8f-8d0e-3013c15902a2)

## X and Y values

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/35b54f3c-ee91-4b3b-ae5c-a816f34112a6)

## Predication values of X and Y

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/81bf6a5f-3990-4f28-bf2d-e42dcf1035f4)

## MSE,MAE and RMSE

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/cac82b43-0bcc-4b2a-9127-2b0b6a4ffefb)

## Training Set

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/8992a768-3f7c-4cf6-88f6-ad64e849935b)

## Testing Set

![image](https://github.com/SamyukthaSreenivasan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475703/54634065-7945-4c7d-8ac9-8080f397a76a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
