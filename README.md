# Task 1 - Prediction using supervised Machine Learnig 
# Author : Amal Madhu 


# prediction-using-supervised-machine-learning-
#Here I am trying to make prediction using supervised machine learning 

# Importing libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')



# reading the data from remote link 

url = "http://bit.ly/w-data"   
data = pd.read_csv(url)
data.head()


# plotting the distribution of scores 

data.plot(x = "Hours", y = "Scores", style = "o")
plt.title("Hours vs Percentage ")
plt.xlabel("Hours Studied ")
plt.ylabel("Percentage Scores")
plt.show()


# The graph is showing a clear positive linear relation between the number of hours studied and percentage of score


# preparing the data 

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values



# splitting the data into training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                   test_size = 0.2, random_state = 0)


# training the algorithm 

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_train, y_train)



# plotting the regression line 

line = regressor.coef_*x + regressor.intercept_

# plotting for the test data 

plt.scatter(x,y)
plt.plot(x, line);
plt.show()


# making prediction 

print(x_test)
y_pred = regressor.predict(x_test)




# comparing actual and predicted values 

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# Predicting using our own values 

hours = 9.25
test = np.array([hours])
test = test.reshape(-1,1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

# Evaluating the model 

from sklearn import metrics
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean squared error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Successfully compledted the prediction using supervised machine leaning task. 
