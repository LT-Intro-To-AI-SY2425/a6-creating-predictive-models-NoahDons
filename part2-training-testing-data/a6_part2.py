import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Create your training and testing datasets:

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=.2)

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)
xtrain = xtrain.reshape(-1,1)

# Create the model
model = LinearRegression().fit(xtrain,ytrain)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coefficient = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x,y)

# Print out the linear equation and r squared value:

print("Linear Equation: y=", coefficient, "x +", intercept)
print("r squared:", r_squared)
'''
**********TEST THE MODEL**********
'''
# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1,1)

# get the predicted y values for the xtest values - returns an array of the results

predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
    print("x value:", float(x_coord[0]), "Predicted y value:", predicted_y, "Actual y value:", actual)

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''

plt.figure(figsize=(8,6))
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")

plt.scatter(xtrain, ytrain, c="blue", label="Training Data")
plt.scatter(xtest, ytest, c="red", label="Testing Data")
plt.scatter(xtest, predict, c="orange", label="Predictions")

plt.plot(x, coefficient*x + intercept, c = "green", label = "Line of Best Fit")
plt.legend()
plt.show()