import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values

#split the data into training and testing data
x_train, x_test, y_train, y_test, = train_test_split(x,y, test_size=.2)

#create linear regression model

model = LinearRegression().fit(x_train, y_train)

#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x_train, y_train), 2)

print("coef: ", coef)
print("r squared:", r_squared)

#Loop through the data and print out the predicted prices and the 
#actual prices
predict = model.predict(x_test)
predict = np.around(predict, 2)

print("***************")
print("Testing Results")

for index in range(len(x_test)):
    actual = y_test[index]
    predicted_y = predict[index]
    x_coord = x_test[index]
    print("predicted price:", predicted_y)
    print("actual price:", actual)

x_test[0][0]=150000
x_test[0][1]=20
pred=model.predict(x_test)
print(pred)
