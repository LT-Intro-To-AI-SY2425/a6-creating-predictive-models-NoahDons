import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y

print("X values:", x)
print("Y values:", y)
# Step 2: Standardize the data using StandardScaler, 

scaler = StandardScaler().fit(x)

# Step 3: Transform the data
x = scaler.transform(x)
# Step 4: Split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2)

# Step 5: Fit the data

# Step 6: Create a LogsiticRegression object and fit the data
model = linear_model.LogisticRegression().fit(x_train, y_train)

# Step 7: Print the score to see the accuracy of the model

print("Score:", model.score(x_test, y_test))
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
fails = 0
tests = 0
print("efoihwwhghew", x_test)
for index in range(len(x_test)):
    
    x = x_test[index]
    x = x.reshape(-1, 3)
    
    

    y_pred = int(model.predict(x))
    if y_pred == 0:
        y_pred = "Male"
    else:
        y_pred = "Female"

    actual = y_test[index]
    if actual == 0:
        actual = "Male"
    else:
        actual = "Female"

    print("actual:", actual)
    print("predicted:", y_pred)
    tests+=1
    if actual != y_pred:
        fails+=1

print(fails)
print(tests)
